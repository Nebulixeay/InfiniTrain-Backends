// 逐算子对拍 + 比较器自检 + 探针用例（001-P2 v2 R01/R02/R05/R06）。
//
// 判据（R01 修正后）：
//   1) 先按显式期望分类：ref 为 NaN/±Inf 时要求 got 同类（NaN↔NaN、同号 Inf），
//      且只有声明"允许特殊参考"的域用例才接受 ref 非有限；
//   2) ref 有限时 got 必须有限，再用 |got-ref| <= atol + rtol*|ref|；
//   3) 普通用例出现非预期 NaN/Inf（哪怕两侧相同）一律失败。
//
// 模式（解析在 backend 注册与设备初始化之前完成）：
//   （默认）            设备对拍套件；有失败 → 退出 1
//   --comparator-selftest  纯 host，逐个断言比较器判断，全部符合才退出 0
//   --inject-finite     设备对拍并注入有限误差，预期整轮失败（退出非 0，由外层脚本检查）
//   --inject-nan        设备对拍并注入非预期 NaN，预期整轮失败
//   --probe-case NAME   设备上的探针用例（只分配+拷贝，不发 kernel），配合 MLU_DEBUG_CHECK=1 由脚本断言日志
//   --bf16-gate         028：BF16 公式级门禁（Cast 双向 / Linear fwd+bwd / elementwise promote /
//                       BF16 grad + FP32 param 的 AccumulateGrad），四组全部符合才退出 0
//   --device N          选择设备

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train_mlu/backend.h"
#include "runtime/mlu_debug.h"

namespace {

using infini_train::DataType;
using infini_train::Device;
using infini_train::Tensor;

constexpr cnrtRet_t CNRT_RET_SUCCESS_LOCAL = cnrtSuccess;
constexpr auto kCPU = Device::DeviceType::kCPU;
constexpr auto kMLU = Device::DeviceType::kPrivateUse1;

bool g_failed = false;
bool g_inject_finite = false;
bool g_inject_nan = false;
int g_device = 0;

Device Cpu() { return Device(kCPU, 0); }
Device Mlu() { return Device(kMLU, g_device); }

std::shared_ptr<Tensor> MakeTensor(const std::vector<int64_t> &dims, const std::vector<float> &host, Device dev) {
    auto t = std::make_shared<Tensor>(dims, DataType::kFLOAT32, dev);
    if (dev.type() == kCPU) {
        std::memcpy(t->DataPtr(), host.data(), host.size() * sizeof(float));
        return t;
    }
    infini_train::core::GetDeviceGuardImpl(dev.type())
        ->Memcpy(t->DataPtr(), host.data(), host.size() * sizeof(float), infini_train::core::MemcpyKind::kH2D);
    return t;
}

std::shared_ptr<Tensor> MakeTensorI64(const std::vector<int64_t> &dims, const std::vector<int64_t> &host, Device dev) {
    auto t = std::make_shared<Tensor>(dims, DataType::kINT64, dev);
    if (dev.type() == kCPU) {
        std::memcpy(t->DataPtr(), host.data(), host.size() * sizeof(int64_t));
        return t;
    }
    infini_train::core::GetDeviceGuardImpl(dev.type())
        ->Memcpy(t->DataPtr(), host.data(), host.size() * sizeof(int64_t), infini_train::core::MemcpyKind::kH2D);
    return t;
}

std::vector<float> ToHost(const std::shared_ptr<Tensor> &t) {
    std::vector<float> host(t->NumElements());
    if (t->GetDevice().type() == kCPU) {
        std::memcpy(host.data(), t->DataPtr(), host.size() * sizeof(float));
        return host;
    }
    infini_train::core::GetDeviceGuardImpl(t->GetDevice().type())
        ->Memcpy(host.data(), t->DataPtr(), host.size() * sizeof(float), infini_train::core::MemcpyKind::kD2H);
    return host;
}

std::vector<float> RandVec(size_t n, float scale, uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, scale);
    std::vector<float> v(n);
    for (auto &x : v) { x = dis(gen); }
    return v;
}

// ---------------------------------------------------------------------------
// 比较器（R01）：非有限优先分类，只有两侧都有限时才用 atol+rtol
// ---------------------------------------------------------------------------
struct Verdict {
    size_t bad = 0;
    size_t worst = 0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    bool empty() const { return bad == 0; }
};

Verdict Evaluate(const std::vector<float> &ref, const std::vector<float> &got, double atol, double rtol,
                 bool allow_special_ref) {
    Verdict v;
    if (ref.size() != got.size()) {
        v.bad = 1;
        return v;
    }
    for (size_t i = 0; i < ref.size(); ++i) {
        const double a = ref[i], b = got[i];
        const bool a_nan = std::isnan(a), b_nan = std::isnan(b);
        const bool a_inf = std::isinf(a), b_inf = std::isinf(b);

        if (a_nan || a_inf) {
            // 参考本身非有限：只有声明允许时才可接受，且必须同类同号
            const bool same = (a_nan && b_nan) || (a_inf && b_inf && ((a > 0) == (b > 0)));
            if (!allow_special_ref || !same) {
                ++v.bad;
                v.worst = i;
                v.max_abs = std::numeric_limits<double>::infinity();
            }
            continue;
        }
        if (b_nan || b_inf) { // 参考有限但结果非有限：一律失败
            ++v.bad;
            v.worst = i;
            v.max_abs = std::numeric_limits<double>::infinity();
            continue;
        }
        const double d = std::fabs(a - b);
        const double tol = atol + rtol * std::fabs(a);
        const double r = d / (std::fabs(a) > 1e-12 ? std::fabs(a) : 1e-12);
        if (d > tol) {
            ++v.bad;
        }
        if (d > v.max_abs) {
            v.max_abs = d;
            v.worst = i;
        }
        if (r > v.max_rel) {
            v.max_rel = r;
        }
    }
    return v;
}

bool Compare(const std::string &name, const std::vector<float> &ref, const std::vector<float> &got, double atol,
             double rtol, bool allow_special_ref = false) {
    const Verdict v = Evaluate(ref, got, atol, rtol, allow_special_ref);
    const bool ok = v.empty();
    if (!ok) {
        g_failed = true;
    }
    std::printf("%-42s %-5s n=%-6zu max_abs=%-11.3e max_rel=%-11.3e bad=%-3zu worst[%zu] ref=%.7g got=%.7g\n",
                name.c_str(), ok ? "PASS" : "FAIL", ref.size(), v.max_abs, v.max_rel, v.bad, v.worst,
                ref.empty() ? 0.0f : ref[v.worst], got.empty() ? 0.0f : got[v.worst]);
    return ok;
}

// 用解析期望值直接断言（不依赖 CPU 实现，R06）
bool CompareAnalytic(const std::string &name, const std::vector<double> &expect, const std::vector<float> &got,
                     double atol, double rtol) {
    std::vector<float> ref(expect.size());
    for (size_t i = 0; i < expect.size(); ++i) { ref[i] = static_cast<float>(expect[i]); }
    return Compare(name, ref, got, atol, rtol, /*allow_special_ref=*/true);
}

// ---------------------------------------------------------------------------
// 纯 host：比较器自检（R02/R06），不初始化设备
// ---------------------------------------------------------------------------
int RunComparatorSelftest() {
    struct Case {
        const char *name;
        std::vector<float> ref;
        std::vector<float> got;
        bool allow_special;
        bool expect_pass;
    };
    const float inf = std::numeric_limits<float>::infinity();
    const float nan_v = std::numeric_limits<float>::quiet_NaN();
    const std::vector<Case> cases = {
        {"finite equal", {1.0f}, {1.0f}, false, true},
        {"finite within tol", {1.0f}, {1.0f + 5e-5f}, false, true},
        {"finite beyond tol", {1.0f}, {1.01f}, false, false},
        {"ref +Inf got 0", {inf}, {0.0f}, false, false},
        {"ref -Inf got 0", {-inf}, {0.0f}, false, false},
        {"ref +Inf got -Inf", {inf}, {-inf}, false, false},
        {"ref +Inf got +Inf allowed", {inf}, {inf}, true, true},
        {"ref +Inf got +Inf not allowed", {inf}, {inf}, false, false},
        {"ref NaN got NaN not allowed", {nan_v}, {nan_v}, false, false},
        {"ref NaN got NaN allowed", {nan_v}, {nan_v}, true, true},
        {"ref NaN got 1 allowed", {nan_v}, {1.0f}, true, false},
        {"ref finite got NaN", {1.0f}, {nan_v}, true, false},
        {"ref finite got Inf", {1.0f}, {inf}, true, false},
        {"shape mismatch", {1.0f, 2.0f}, {1.0f}, false, false},
        {"tolerance boundary just inside", {2.0f}, {2.0f + 1e-5f + 1e-4f * 2.0f * 0.99f}, false, true},
        {"tolerance boundary just outside", {2.0f}, {2.0f + 1e-5f + 1e-4f * 2.0f * 1.5f}, false, false},
    };
    int failures = 0;
    for (const auto &c : cases) {
        const Verdict v = Evaluate(c.ref, c.got, 1e-5, 1e-4, c.allow_special);
        const bool pass = v.empty();
        const bool ok = (pass == c.expect_pass);
        if (!ok) {
            ++failures;
        }
        std::printf("%-38s expect=%-4s actual=%-4s %s\n", c.name, c.expect_pass ? "PASS" : "FAIL",
                    pass ? "PASS" : "FAIL", ok ? "ok" : "MISMATCH");
    }
    std::printf("comparator-selftest: %s (%d mismatch)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// 设备侧小算子用例
// ---------------------------------------------------------------------------
void ApplyInjection(std::vector<float> &mlu) {
    if (mlu.empty()) {
        return;
    }
    if (g_inject_finite) {
        mlu[0] += 1.0f;
    }
    if (g_inject_nan) {
        mlu[0] = std::numeric_limits<float>::quiet_NaN();
    }
}

void TestLinearForward(int64_t rows, int64_t in_f, int64_t out_f, bool transpose, bool with_bias) {
    auto x = RandVec(static_cast<size_t>(rows * in_f), 1.0f, 101);
    auto w = RandVec(static_cast<size_t>(in_f * out_f), 0.05f, 102);
    auto b = RandVec(static_cast<size_t>(out_f), 0.02f, 103);
    const std::vector<int64_t> x_dims{rows, in_f};
    const std::vector<int64_t> w_dims
        = transpose ? std::vector<int64_t>{out_f, in_f} : std::vector<int64_t>{in_f, out_f};
    auto run = [&](Device dev) {
        auto xt = MakeTensor(x_dims, x, dev);
        auto wt = MakeTensor(w_dims, w, dev);
        auto bt = with_bias ? MakeTensor({out_f}, b, dev) : nullptr;
        auto out = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({dev.type(), "LinearForward"}, xt,
                                                                                      wt, transpose, bt);
        return ToHost(out);
    };
    char name[128];
    std::snprintf(name, sizeof(name), "Linear fwd rows=%lld in=%lld out=%lld T=%d bias=%d",
                  static_cast<long long>(rows), static_cast<long long>(in_f), static_cast<long long>(out_f),
                  static_cast<int>(transpose), static_cast<int>(with_bias));
    auto cpu = run(Cpu());
    auto mlu = run(Mlu());
    ApplyInjection(mlu);
    Compare(name, cpu, mlu, 1e-5, 1e-4);
}

void TestLinearBackward(int64_t rows, int64_t in_f, int64_t out_f, bool transpose) {
    auto x = RandVec(static_cast<size_t>(rows * in_f), 1.0f, 201);
    auto w = RandVec(static_cast<size_t>(in_f * out_f), 0.05f, 202);
    auto g = RandVec(static_cast<size_t>(rows * out_f), 0.5f, 203);
    const std::vector<int64_t> x_dims{rows, in_f};
    const std::vector<int64_t> w_dims
        = transpose ? std::vector<int64_t>{out_f, in_f} : std::vector<int64_t>{in_f, out_f};
    const std::vector<int64_t> g_dims{rows, out_f};
    auto run = [&](Device dev) {
        auto wt = MakeTensor(w_dims, w, dev);
        auto gt = MakeTensor(g_dims, g, dev);
        auto xt = MakeTensor(x_dims, x, dev);
        auto dx = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
            {dev.type(), "LinearBackwardInput"}, wt, gt, transpose, in_f, out_f, x_dims);
        auto dw = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
            {dev.type(), "LinearBackwardWeight"}, xt, gt, transpose, in_f, out_f);
        auto db = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({dev.type(), "LinearBackwardBias"},
                                                                                     gt, out_f);
        return std::make_tuple(ToHost(dx), ToHost(dw), ToHost(db));
    };
    auto [cpu_dx, cpu_dw, cpu_db] = run(Cpu());
    auto [mlu_dx, mlu_dw, mlu_db] = run(Mlu());
    ApplyInjection(mlu_dx);
    char tag[96];
    std::snprintf(tag, sizeof(tag), "rows=%lld in=%lld out=%lld T=%d", static_cast<long long>(rows),
                  static_cast<long long>(in_f), static_cast<long long>(out_f), static_cast<int>(transpose));
    Compare(std::string("Linear bwd dx ") + tag, cpu_dx, mlu_dx, 1e-5, 1e-4);
    Compare(std::string("Linear bwd dw ") + tag, cpu_dw, mlu_dw, 1e-5, 1e-4);
    Compare(std::string("Linear bwd db ") + tag, cpu_db, mlu_db, 1e-5, 1e-4);
}

// Pow：解析期望（R06）——0^0.5=0, 0^0=1, (-2)^9=-512, (-2)^0.5=NaN, 2^0.5=sqrt2,
// (-2)^3=-8, 0.5^-1=2, base=3/exp=2 => 9
void TestPowAnalytic() {
    struct Case {
        float input;
        float exponent;
        bool base_is_scalar;
        double expect;
        const char *desc;
    };
    const Case cases[] = {
        {0.0f, 0.5f, false, 0.0, "0^0.5"},
        {0.0f, 0.0f, false, 1.0, "0^0"},
        {-2.0f, 9.0f, false, -512.0, "(-2)^9"},
        {-2.0f, 0.5f, false, std::numeric_limits<double>::quiet_NaN(), "(-2)^0.5"},
        {2.0f, 0.5f, false, std::sqrt(2.0), "2^0.5"},
        {-2.0f, 3.0f, false, -8.0, "(-2)^3"},
        {0.5f, -1.0f, false, 2.0, "0.5^-1"},
        {2.0f, 3.0f, true, 9.0, "base=3 exp=2 -> 9"}, // tensor=2 是指数，scalar=3 是底数
    };
    for (const auto &c : cases) {
        auto run = [&](Device dev) {
            auto xt = MakeTensor({1}, {c.input}, dev);
            auto out = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
                {dev.type(), "PowForward"}, xt, c.exponent, c.base_is_scalar);
            return ToHost(out);
        };
        auto cpu = run(Cpu());
        auto got = run(Mlu());
        char name[96];
        std::snprintf(name, sizeof(name), "Pow %s", c.desc);
        // 解析期望与 CPU 参考双保险：先按解析值断言，再与 CPU 对拍
        CompareAnalytic(std::string(name) + " [analytic]", {c.expect}, got, 1e-6, 1e-5);
        Compare(std::string(name) + " [vs cpu]", cpu, got, 1e-6, 1e-5, /*allow_special_ref=*/true);
    }

    // scalar_is_base 反向：y=base^x, dy/dx = base^x*ln(base)；base=3, x=2, grad=2 -> 2*9*ln3
    {
        const float base = 3.0f, xval = 2.0f, grad = 2.0f;
        const double expect = static_cast<double>(grad) * std::pow(base, xval) * std::log(base);
        auto run = [&](Device dev) {
            auto gt = MakeTensor({1}, {grad}, dev);
            auto xt = MakeTensor({1}, {xval}, dev);
            auto out = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({dev.type(), "PowBackward"},
                                                                                          gt, xt, base, true);
            return ToHost(out);
        };
        CompareAnalytic("Pow bwd scalar_is_base=3 x=2 grad=2", {expect}, run(Mlu()), 1e-5, 1e-5);
    }

    // Pow(x,3) 前向/反向，输入显式含 0 与负值
    {
        const std::vector<float> xs = {0.0f, -2.0f, 0.5f, 3.0f};
        const std::vector<float> gs = {1.0f, 2.0f, -1.0f, 0.5f};
        std::vector<double> exp_f(4), exp_b(4);
        for (size_t i = 0; i < xs.size(); ++i) {
            const double x = xs[i];
            exp_f[i] = x * x * x;
            exp_b[i] = static_cast<double>(gs[i]) * 3.0 * x * x; // x=0 处为 0
        }
        auto run_fwd = [&](Device dev) {
            auto xt = MakeTensor({4}, xs, dev);
            auto out = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({dev.type(), "PowForward"},
                                                                                          xt, 3.0f, false);
            return ToHost(out);
        };
        auto run_bwd = [&](Device dev) {
            auto gt = MakeTensor({4}, gs, dev);
            auto xt = MakeTensor({4}, xs, dev);
            auto out = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({dev.type(), "PowBackward"},
                                                                                          gt, xt, 3.0f, false);
            return ToHost(out);
        };
        CompareAnalytic("Pow(x,3) fwd incl x=0", exp_f, run_fwd(Mlu()), 1e-6, 1e-5);
        CompareAnalytic("Pow(x,3) bwd incl x=0", exp_b, run_bwd(Mlu()), 1e-6, 1e-5);
    }
}

void TestCrossEntropy(int64_t bs, int64_t classes) {
    auto logits = RandVec(static_cast<size_t>(bs * classes), 1.0f, 401);
    std::vector<int64_t> target(static_cast<size_t>(bs));
    for (int64_t i = 0; i < bs; ++i) { target[i] = i % classes; }
    auto run = [&](Device dev) {
        auto lt = MakeTensor({bs, classes}, logits, dev);
        auto tt = MakeTensorI64({bs}, target, dev);
        auto out = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
            {dev.type(), "CrossEntropyForward"}, lt, tt);
        return ToHost(out);
    };
    Compare("CrossEntropy fwd (bs=4,cls=8)", run(Cpu()), run(Mlu()), 1e-5, 1e-4);
}

void TestLayerNorm(int64_t rows, int64_t dim) {
    auto x = RandVec(static_cast<size_t>(rows * dim), 1.0f, 11);
    auto w = RandVec(static_cast<size_t>(dim), 0.1f, 12);
    auto b = RandVec(static_cast<size_t>(dim), 0.1f, 13);
    auto run = [&](Device dev) {
        auto xt = MakeTensor({rows, 1, dim}, x, dev);
        auto wt = MakeTensor({dim}, w, dev);
        auto bt = MakeTensor({dim}, b, dev);
        auto res = infini_train::Dispatcher::Instance()
                       .Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
                           {dev.type(), "LayerNormForward"}, xt, wt, bt, 1e-5f);
        return ToHost(std::get<0>(res));
    };
    Compare("LayerNorm", run(Cpu()), run(Mlu()), 1e-5, 1e-4);
}

void TestSoftmax(int64_t rows, int64_t cols) {
    auto x = RandVec(static_cast<size_t>(rows * cols), 1.0f, 21);
    auto run = [&](Device dev) {
        auto xt = MakeTensor({rows, cols}, x, dev);
        auto out = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({dev.type(), "SoftmaxForward"},
                                                                                      xt, int64_t{-1});
        return ToHost(out);
    };
    Compare("Softmax(dim=-1)", run(Cpu()), run(Mlu()), 1e-5, 1e-4);
}

void TestAddBroadcast(int64_t rows, int64_t cols) {
    auto a = RandVec(static_cast<size_t>(rows * cols), 1.0f, 31);
    auto b = RandVec(static_cast<size_t>(cols), 0.5f, 32);
    auto run = [&](Device dev) {
        auto at = MakeTensor({rows, cols}, a, dev);
        auto bt = MakeTensor({cols}, b, dev);
        auto out
            = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({dev.type(), "AddForward"}, at, bt);
        return ToHost(out);
    };
    Compare("Add(broadcast)", run(Cpu()), run(Mlu()), 1e-5, 1e-4);
}

void TestMulScalar(int64_t n, float s) {
    auto x = RandVec(static_cast<size_t>(n), 1.0f, 41);
    auto run = [&](Device dev) {
        auto xt = MakeTensor({n}, x, dev);
        auto out = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({dev.type(), "MulScalarForward"},
                                                                                      xt, s);
        return ToHost(out);
    };
    auto cpu = run(Cpu());
    auto mlu = run(Mlu());
    ApplyInjection(mlu);
    Compare("MulScalar", cpu, mlu, 1e-5, 1e-4);
}

void TestTranspose(int64_t b, int64_t h, int64_t t, int64_t d) {
    auto x = RandVec(static_cast<size_t>(b * h * t * d), 1.0f, 51);
    auto run = [&](Device dev) {
        auto xt = MakeTensor({b, h, t, d}, x, dev);
        auto out = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({dev.type(), "TransposeForward"},
                                                                                      xt, int64_t{1}, int64_t{2});
        return ToHost(out);
    };
    Compare("Transpose(1,2)", run(Cpu()), run(Mlu()), 1e-5, 1e-4);
}

// ---------------------------------------------------------------------------
// 探针用例（R05）：只分配 + 拷贝，不发 kernel；由脚本配合 MLU_DEBUG_CHECK=1 断言日志
// ---------------------------------------------------------------------------
template <typename Bits>
std::shared_ptr<Tensor> MakeRawTensor(const std::vector<int64_t> &dims, const std::vector<Bits> &bits, DataType dt,
                                      Device dev) {
    auto t = std::make_shared<Tensor>(dims, dt, dev);
    infini_train::core::GetDeviceGuardImpl(dev.type())
        ->Memcpy(t->DataPtr(), bits.data(), bits.size() * sizeof(Bits), infini_train::core::MemcpyKind::kH2D);
    return t;
}

int RunProbeCase(const std::string &name) {
    using infini_train::core::mlu::DebugCheck;
    if (name == "fp32_special") {
        // [1, NaN, +Inf, -Inf, 0] -> non_finite=3, 有限 min=0 max=1
        auto t = MakeTensor({5},
                            {1.0f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity(), 0.0f},
                            Mlu());
        DebugCheck<float>(t, "probe_fp32_special");
    } else if (name == "first_nan") {
        // 首元素 NaN：[NaN, 1, 2] -> non_finite=1, min=1 max=2（验证首个有限值初始化）
        auto t = MakeTensor({3}, {std::numeric_limits<float>::quiet_NaN(), 1.0f, 2.0f}, Mlu());
        DebugCheck<float>(t, "probe_first_nan");
    } else if (name == "all_nonfinite") {
        auto t = MakeTensor({3},
                            {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()},
                            Mlu());
        DebugCheck<float>(t, "probe_all_nonfinite");
    } else if (name == "bf16_bits") {
        // 0x7c00 是合法有限 bf16（指数 0xf8），0x7f80 才是 +Inf，0x3f80=1.0，0x8000=-0
        std::vector<uint16_t> bits = {0x7c00, 0x7f80, 0x3f80, 0x8000};
        auto t = MakeRawTensor<uint16_t>({4}, bits, DataType::kBFLOAT16, Mlu());
        DebugCheck<uint16_t>(t, "probe_bf16_bits");
    } else if (name == "fp16_bits") {
        // 0x7c00 是 +Inf（fp16 指数 0x1f），0x3c00=1.0，0xfc00=-Inf
        std::vector<uint16_t> bits = {0x7c00, 0x3c00, 0xfc00};
        auto t = MakeRawTensor<uint16_t>({3}, bits, DataType::kFLOAT16, Mlu());
        DebugCheck<uint16_t>(t, "probe_fp16_bits");
    } else if (name == "int_wide") {
        // 宽整数范围（R04）：UINT64_MAX 与 INT64_MIN/MAX 不得显示成 -1 或溢出
        std::vector<uint64_t> u = {0, 18446744073709551615ULL};
        auto tu = MakeRawTensor<uint64_t>({2}, u, DataType::kUINT64, Mlu());
        DebugCheck<uint64_t>(tu, "probe_uint64");
        std::vector<int64_t> s = {INT64_MIN, INT64_MAX};
        auto ts = MakeRawTensor<int64_t>({2}, s, DataType::kINT64, Mlu());
        DebugCheck<int64_t>(ts, "probe_int64");
    } else {
        std::printf("unknown probe case: %s\n", name.c_str());
        return 2;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// 007：关键反向设备用例（--backward-group a|b|all）
//
// 同步约定：所有 D2H 一律用既有阻塞 ToHost（MluGuardImpl::Memcpy：在当前 queue 上
// cnrtMemcpyAsync + cnrtQueueSync，函数返回即表示已完成；它返回 void，内部用
// CNRT_CHECK 检查，失败会终止进程）。禁止用 Tensor::To(kCPU)：tensor.cc 的 D2H
// 只入队不同步，不能当完成证明。每个 MLU case 内显式 DeviceGuard(Mlu())，保证
// current device/stream 与 kernel 用的是同一队列。
// ---------------------------------------------------------------------------
std::string g_backward_group;

bool BwdCheckTensor(const char *name, const std::shared_ptr<Tensor> &t, const std::vector<int64_t> &dims,
                    DataType dtype) {
    if (!t) {
        std::printf("%-46s FAIL null tensor\n", name);
        g_failed = true;
        return false;
    }
    if (t->Dims() != dims) {
        std::printf("%-46s FAIL unexpected shape\n", name);
        g_failed = true;
        return false;
    }
    if (t->Dtype() != dtype) {
        std::printf("%-46s FAIL unexpected dtype\n", name);
        g_failed = true;
        return false;
    }
    return true;
}

// 同步日志 helper：必须在 DeviceGuard(Mlu()) 内调用，取到的才是本 case 实际使用的 stream。
std::string MluStreamTag() {
    auto *impl = infini_train::core::GetDeviceGuardImpl(Device::DeviceType::kPrivateUse1);
    auto *stream = impl->GetStream(Mlu());
    char buf[128];
    std::snprintf(buf, sizeof(buf), "device=%d stream=%p", g_device, static_cast<void *>(stream));
    return std::string(buf);
}

// 必须在对应 D2H 阻塞读回返回之后才调用。返回 void 的是内部 MluGuardImpl::Memcpy
// （当前 queue cnrtMemcpyAsync + cnrtQueueSync，失败由 CNRT_CHECK 终止进程）；
// ToHost 本身返回 std::vector<float>，本日志只声明"该次读回已返回"，不伪造 Memcpy 的 rc。
void LogBackwardCaseSync(const char *name) {
    std::printf("[bwd] case=%s %s sync=blocking_Memcpy_current_queue host_read_after_tohost_returned "
                "internal_check=CNRT_CHECK\n",
                name, MluStreamTag().c_str());
}

// 先验 size/有限性，再走既有 Compare（rtol=0，只用绝对门槛）；注入只改结果副本。
bool BwdCompare(const char *name, const std::vector<double> &expect, const std::vector<float> &got, double atol) {
    if (expect.size() != got.size()) {
        std::printf("%-46s FAIL size expect=%zu got=%zu\n", name, expect.size(), got.size());
        g_failed = true;
        return false;
    }
    std::vector<float> ref(expect.size());
    for (size_t i = 0; i < expect.size(); ++i) {
        if (!std::isfinite(expect[i])) {
            std::printf("%-46s FAIL non-finite analytic reference at %zu\n", name, i);
            g_failed = true;
            return false;
        }
        ref[i] = static_cast<float>(expect[i]);
        if (!std::isfinite(ref[i])) { // double -> float 转换溢出
            std::printf("%-46s FAIL analytic reference overflows float at %zu\n", name, i);
            g_failed = true;
            return false;
        }
    }
    std::vector<float> got_injected = got;
    ApplyInjection(got_injected);
    return Compare(name, ref, got_injected, atol, 0.0, /*allow_special_ref=*/false);
}

std::vector<double> ToDouble(const std::vector<float> &v) { return std::vector<double>(v.begin(), v.end()); }

void TestEmbeddingBackward007() {
    const std::vector<int64_t> indices{0, 3, 3, 7, 0, 7, 7, 1};
    const int64_t tokens = static_cast<int64_t>(indices.size());
    const int64_t vocab = 32;
    const int64_t dim = 16;
    const std::vector<int64_t> weight_dims{vocab, dim};
    std::vector<float> grad(static_cast<size_t>(tokens * dim));
    for (size_t i = 0; i < grad.size(); ++i) { grad[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 64.0f; }
    // 独立 double scatter-add 参考（含重复 token 累加）
    std::vector<double> expect(static_cast<size_t>(vocab * dim), 0.0);
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t c = 0; c < dim; ++c) {
            expect[static_cast<size_t>(indices[static_cast<size_t>(t)] * dim + c)]
                += static_cast<double>(grad[static_cast<size_t>(t * dim + c)]);
        }
    }

    auto in_cpu = MakeTensorI64({tokens}, indices, Cpu());
    auto g_cpu = MakeTensor({tokens, dim}, grad, Cpu());
    auto cpu = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kCPU, "EmbeddingBackward"}, in_cpu,
                                                                                  weight_dims, g_cpu);
    if (!BwdCheckTensor("Embedding bwd cpu shape/dtype", cpu, weight_dims, DataType::kFLOAT32)) {
        return;
    }
    std::vector<float> cpu_host = ToHost(cpu);

    infini_train::core::DeviceGuard guard(Mlu());
    auto in_mlu = MakeTensorI64({tokens}, indices, Mlu());
    auto g_mlu = MakeTensor({tokens, dim}, grad, Mlu());
    auto mlu = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kMLU, "EmbeddingBackward"}, in_mlu,
                                                                                  weight_dims, g_mlu);
    if (!BwdCheckTensor("Embedding bwd shape/dtype", mlu, weight_dims, DataType::kFLOAT32)) {
        return;
    }
    std::vector<float> mlu_host = ToHost(mlu);
    LogBackwardCaseSync("EmbeddingBackward");
    BwdCompare("Embedding bwd vs analytic", expect, mlu_host, 1e-6);
    BwdCompare("Embedding bwd vs cpu", ToDouble(cpu_host), mlu_host, 1e-6);

    const int64_t used_rows[] = {0, 1, 3, 7};
    bool unused_zero = true;
    for (int64_t row = 0; row < vocab; ++row) {
        bool used = false;
        for (int64_t u : used_rows) {
            if (u == row) {
                used = true;
            }
        }
        if (used) {
            continue;
        }
        for (int64_t c = 0; c < dim; ++c) {
            if (mlu_host[static_cast<size_t>(row * dim + c)] != 0.0f) {
                unused_zero = false;
            }
        }
    }
    std::printf("%-46s %s\n", "Embedding bwd unused rows are zero", unused_zero ? "PASS" : "FAIL");
    if (!unused_zero) {
        g_failed = true;
    }
}

void TestCrossEntropyBackward007(int64_t rows) {
    const int64_t classes = 32;
    std::vector<float> logits(static_cast<size_t>(rows * classes));
    for (size_t i = 0; i < logits.size(); ++i) {
        logits[i] = static_cast<float>(static_cast<int>(i % 23) - 11) / 16.0f;
    }
    std::vector<int64_t> target(static_cast<size_t>(rows));
    for (int64_t r = 0; r < rows; ++r) { target[static_cast<size_t>(r)] = (3 * r + 1) % classes; }
    const float grad_scalar = 0.5f;
    const double bs = static_cast<double>(rows);
    std::vector<double> expect(logits.size(), 0.0);
    for (int64_t r = 0; r < rows; ++r) {
        double max_logit = -std::numeric_limits<double>::infinity();
        for (int64_t c = 0; c < classes; ++c) {
            max_logit = std::max(max_logit, static_cast<double>(logits[static_cast<size_t>(r * classes + c)]));
        }
        double sum_exp = 0.0;
        for (int64_t c = 0; c < classes; ++c) {
            sum_exp += std::exp(static_cast<double>(logits[static_cast<size_t>(r * classes + c)]) - max_logit);
        }
        for (int64_t c = 0; c < classes; ++c) {
            const double p
                = std::exp(static_cast<double>(logits[static_cast<size_t>(r * classes + c)]) - max_logit) / sum_exp;
            expect[static_cast<size_t>(r * classes + c)]
                = (p - (c == target[static_cast<size_t>(r)] ? 1.0 : 0.0)) * static_cast<double>(grad_scalar) / bs;
        }
    }

    const std::vector<int64_t> shape{rows, classes};
    auto in_cpu = MakeTensor(shape, logits, Cpu());
    auto t_cpu = MakeTensorI64({rows}, target, Cpu());
    auto g_cpu = MakeTensor({}, {grad_scalar}, Cpu());
    auto cpu = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kCPU, "CrossEntropyBackward"},
                                                                                  in_cpu, t_cpu, g_cpu);
    if (!BwdCheckTensor("CE bwd cpu shape/dtype", cpu, shape, DataType::kFLOAT32)) {
        return;
    }
    std::vector<float> cpu_host = ToHost(cpu);

    infini_train::core::DeviceGuard guard(Mlu());
    auto in_mlu = MakeTensor(shape, logits, Mlu());
    auto t_mlu = MakeTensorI64({rows}, target, Mlu());
    auto g_mlu = MakeTensor({}, {grad_scalar}, Mlu());
    auto mlu = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kMLU, "CrossEntropyBackward"},
                                                                                  in_mlu, t_mlu, g_mlu);
    if (!BwdCheckTensor("CE bwd shape/dtype", mlu, shape, DataType::kFLOAT32)) {
        return;
    }
    std::vector<float> mlu_host = ToHost(mlu);
    char tag[64];
    std::snprintf(tag, sizeof(tag), "CE bwd rows=%lld", static_cast<long long>(rows));
    LogBackwardCaseSync(tag);
    BwdCompare((std::string(tag) + " vs analytic").c_str(), expect, mlu_host, 1e-5);
    BwdCompare((std::string(tag) + " vs cpu").c_str(), ToDouble(cpu_host), mlu_host, 1e-5);
}

void TestAccumulateGrad007() {
    const int64_t n = 64;
    std::vector<float> init(static_cast<size_t>(n));
    std::vector<float> grad(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        init[static_cast<size_t>(i)] = static_cast<float>(static_cast<int>(i % 9) - 4) / 8.0f;
        if (init[static_cast<size_t>(i)] == 0.0f) { // 契约要求非零初值
            init[static_cast<size_t>(i)] = 0.125f;
        }
        grad[static_cast<size_t>(i)] = static_cast<float>(static_cast<int>(i % 7) - 3) / 16.0f;
        if (grad[static_cast<size_t>(i)] == 0.0f) { // 契约要求非零 gradient
            grad[static_cast<size_t>(i)] = -0.0625f;
        }
    }
    struct Role {
        const char *name;
        float rate;
    };
    const Role roles[]
        = {{"AccumulateGrad grad_accum rate=+0.5", 0.5f}, {"AccumulateGrad sgd_update rate=-0.5", -0.5f}};
    int failures = 0;
    int assertions = 0;
    for (const auto &role : roles) {
        std::vector<double> expect(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            expect[static_cast<size_t>(i)]
                = static_cast<double>(init[static_cast<size_t>(i)])
                + static_cast<double>(role.rate) * static_cast<double>(grad[static_cast<size_t>(i)]);
        }
        {
            infini_train::core::DeviceGuard guard(Mlu());
            auto t_mlu = MakeTensor({n}, init, Mlu());
            auto g_mlu = MakeTensor({n}, grad, Mlu());
            if (!BwdCheckTensor("AccumulateGrad mlu target shape/dtype", t_mlu, {n}, DataType::kFLOAT32)
                || !BwdCheckTensor("AccumulateGrad mlu grad shape/dtype", g_mlu, {n}, DataType::kFLOAT32)) {
                return;
            }
            auto kernel = infini_train::Dispatcher::Instance().GetKernel({kMLU, "AccumulateGrad"});
            kernel.Call<void>(g_mlu, role.rate, t_mlu);
            // 原地张量：调用后重新校验再读回
            if (!BwdCheckTensor("AccumulateGrad mlu target after call", t_mlu, {n}, DataType::kFLOAT32)) {
                return;
            }
            std::vector<float> mlu_host = ToHost(t_mlu);
            LogBackwardCaseSync(role.name);
            ++assertions;
            if (!BwdCompare((std::string(role.name) + " vs analytic").c_str(), expect, mlu_host, 1e-6)) {
                ++failures;
            }

            auto t_cpu = MakeTensor({n}, init, Cpu());
            auto g_cpu = MakeTensor({n}, grad, Cpu());
            if (!BwdCheckTensor("AccumulateGrad cpu target shape/dtype", t_cpu, {n}, DataType::kFLOAT32)
                || !BwdCheckTensor("AccumulateGrad cpu grad shape/dtype", g_cpu, {n}, DataType::kFLOAT32)) {
                return;
            }
            auto cpu_kernel = infini_train::Dispatcher::Instance().GetKernel({kCPU, "AccumulateGrad"});
            cpu_kernel.Call<void>(g_cpu, role.rate, t_cpu);
            // 原地张量：CPU 调用后同样重新校验再读回
            if (!BwdCheckTensor("AccumulateGrad cpu target after call", t_cpu, {n}, DataType::kFLOAT32)) {
                return;
            }
            ++assertions;
            if (!BwdCompare((std::string(role.name) + " vs cpu").c_str(), ToDouble(ToHost(t_cpu)), mlu_host, 1e-6)) {
                ++failures;
            }
        }
    }
    // 中性汇总：不打印 PASS，也不隐瞒上面的失败（g_failed/rc 已由 BwdCompare 设置）
    std::printf("%-46s neutral: assertions=%d failed=%d (kernel-level rate, not the training accumulation rate)\n",
                "AccumulateGrad roles finished", assertions, failures);
}

void TestSoftmaxBackward007() {
    const int64_t rows = 4;
    const int64_t cols = 64;
    const std::vector<int64_t> shape{rows, cols};
    const int64_t dim = 1;
    std::vector<float> base(static_cast<size_t>(rows * cols));
    std::vector<float> grad(static_cast<size_t>(rows * cols));
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            const size_t i = static_cast<size_t>(r * cols + c);
            base[i] = static_cast<float>(static_cast<int>(c % 11) - 5) / 5.0f;
            grad[i] = static_cast<float>(static_cast<int>((c * 3 + r) % 13) - 6) / 8.0f;
        }
    }
    struct Mask {
        const char *name;
        int kind; // 0 finite, 1 -inf, 2 float32 lowest
    };
    const Mask masks[] = {{"Softmax bwd finite", 0}, {"Softmax bwd -inf mask", 1}, {"Softmax bwd lowest mask", 2}};
    const float lowest = std::numeric_limits<float>::lowest();
    for (const auto &mask : masks) {
        std::vector<float> logits = base;
        if (mask.kind != 0) {
            for (int64_t r = 0; r < rows; ++r) {
                for (int64_t c = 0; c < cols; ++c) {
                    if (c % 2 == 1) {
                        logits[static_cast<size_t>(r * cols + c)]
                            = (mask.kind == 1) ? -std::numeric_limits<float>::infinity() : lowest;
                    }
                }
            }
        }
        // 独立 double 稳定 softmax + 反向解析参考
        std::vector<double> expect(static_cast<size_t>(rows * cols), 0.0);
        std::vector<double> prob(static_cast<size_t>(rows * cols), 0.0);
        for (int64_t r = 0; r < rows; ++r) {
            double max_logit = -std::numeric_limits<double>::infinity();
            for (int64_t c = 0; c < cols; ++c) {
                const float v = logits[static_cast<size_t>(r * cols + c)];
                if (std::isfinite(v)) {
                    max_logit = std::max(max_logit, static_cast<double>(v));
                }
            }
            double sum_exp = 0.0;
            for (int64_t c = 0; c < cols; ++c) {
                const float v = logits[static_cast<size_t>(r * cols + c)];
                const double e = std::isfinite(v) ? std::exp(static_cast<double>(v) - max_logit) : 0.0;
                prob[static_cast<size_t>(r * cols + c)] = e;
                sum_exp += e;
            }
            for (int64_t c = 0; c < cols; ++c) { prob[static_cast<size_t>(r * cols + c)] /= sum_exp; }
            double dot = 0.0;
            for (int64_t c = 0; c < cols; ++c) {
                dot += static_cast<double>(grad[static_cast<size_t>(r * cols + c)])
                     * prob[static_cast<size_t>(r * cols + c)];
            }
            for (int64_t c = 0; c < cols; ++c) {
                expect[static_cast<size_t>(r * cols + c)]
                    = prob[static_cast<size_t>(r * cols + c)]
                    * (static_cast<double>(grad[static_cast<size_t>(r * cols + c)]) - dot);
            }
        }

        auto x_cpu = MakeTensor(shape, logits, Cpu());
        auto g_cpu = MakeTensor(shape, grad, Cpu());
        auto y_cpu
            = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kCPU, "SoftmaxForward"}, x_cpu, dim);
        if (!BwdCheckTensor("Softmax cpu fwd shape/dtype", y_cpu, shape, DataType::kFLOAT32)) {
            return;
        }
        std::vector<float> y_cpu_host = ToHost(y_cpu);
        bool y_cpu_finite = true;
        for (float v : y_cpu_host) {
            if (!std::isfinite(v)) {
                y_cpu_finite = false;
            }
        }
        std::printf("%-46s %s\n", (std::string(mask.name) + " cpu forward finite").c_str(),
                    y_cpu_finite ? "PASS" : "FAIL");
        if (!y_cpu_finite) { // CPU 前向非有限：同样记失败并跳过该 mask，不喂给 Backward
            g_failed = true;
            std::printf("%-46s SKIP (cpu forward non-finite; backward not run for this case)\n", mask.name);
            continue;
        }
        auto cpu = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kCPU, "SoftmaxBackward"}, g_cpu,
                                                                                      y_cpu, dim);
        if (!BwdCheckTensor("Softmax cpu bwd shape/dtype", cpu, shape, DataType::kFLOAT32)) {
            return;
        }
        std::vector<float> cpu_host = ToHost(cpu);

        infini_train::core::DeviceGuard guard(Mlu());
        auto x_mlu = MakeTensor(shape, logits, Mlu());
        auto g_mlu = MakeTensor(shape, grad, Mlu());
        auto y_mlu
            = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kMLU, "SoftmaxForward"}, x_mlu, dim);
        if (!BwdCheckTensor("Softmax fwd shape/dtype", y_mlu, shape, DataType::kFLOAT32)) {
            return;
        }
        std::vector<float> y_host = ToHost(y_mlu);
        bool y_finite = true;
        for (float v : y_host) {
            if (!std::isfinite(v)) {
                y_finite = false;
            }
        }
        std::printf("%-46s %s\n", (std::string(mask.name) + " forward finite").c_str(), y_finite ? "PASS" : "FAIL");
        if (!y_finite) {
            g_failed = true;
            std::printf("%-46s SKIP (forward output non-finite; backward not run for this case)\n", mask.name);
            continue;
        }
        if (mask.kind != 0) {
            bool mask_prob_zero = true;
            for (int64_t r = 0; r < rows; ++r) {
                for (int64_t c = 1; c < cols; c += 2) { // 奇数列被掩码
                    const float v = y_host[static_cast<size_t>(r * cols + c)];
                    if (!std::isfinite(v) || v != 0.0f) {
                        mask_prob_zero = false;
                    }
                }
            }
            std::printf("%-46s %s\n", (std::string(mask.name) + " masked prob == 0").c_str(),
                        mask_prob_zero ? "PASS" : "FAIL");
            if (!mask_prob_zero) {
                g_failed = true;
            }
        }
        auto mlu = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kMLU, "SoftmaxBackward"}, g_mlu,
                                                                                      y_mlu, dim);
        if (!BwdCheckTensor("Softmax bwd shape/dtype", mlu, shape, DataType::kFLOAT32)) {
            return;
        }
        std::vector<float> mlu_host = ToHost(mlu);
        LogBackwardCaseSync(mask.name);
        BwdCompare((std::string(mask.name) + " vs analytic").c_str(), expect, mlu_host, 1e-5);
        BwdCompare((std::string(mask.name) + " vs cpu").c_str(), ToDouble(cpu_host), mlu_host, 1e-5);
        if (mask.kind != 0) {
            bool masked_zero = true;
            for (int64_t r = 0; r < rows; ++r) {
                for (int64_t c = 1; c < cols; c += 2) { // 奇数列被掩码
                    const float v = mlu_host[static_cast<size_t>(r * cols + c)];
                    if (!std::isfinite(v) || v != 0.0f) {
                        masked_zero = false;
                    }
                }
            }
            std::printf("%-46s %s\n", (std::string(mask.name) + " masked grad == 0").c_str(),
                        masked_zero ? "PASS" : "FAIL");
            if (!masked_zero) {
                g_failed = true;
            }
        }
    }
}

void TestLayerNormBackward007() {
    const std::vector<int64_t> input_dims{1, 2, 4};
    const std::vector<int64_t> param_dims{4};
    const std::vector<float> input_values{-1.5f, -0.5f, 0.5f, 1.5f, -3.0f, -1.0f, 1.0f, 3.0f};
    const std::vector<float> weight_values{1.0f, 0.5f, -1.0f, 2.0f};
    const std::vector<float> bias_values{0.5f, -0.5f, 1.0f, -1.0f};
    const std::vector<float> grad_values{1.0f, 2.0f, 3.0f, 4.0f, 0.5f, -1.0f, 2.0f, -0.5f};
    const float eps = 1e-5f;

    auto x_cpu = MakeTensor(input_dims, input_values, Cpu());
    auto w_cpu = MakeTensor(param_dims, weight_values, Cpu());
    auto b_cpu = MakeTensor(param_dims, bias_values, Cpu());
    auto g_cpu = MakeTensor(input_dims, grad_values, Cpu());
    auto fwd_cpu = infini_train::Dispatcher::Instance()
                       .Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
                           {kCPU, "LayerNormForward"}, x_cpu, w_cpu, b_cpu, eps);
    if (!BwdCheckTensor("LayerNorm cpu fwd out shape/dtype", std::get<0>(fwd_cpu), input_dims, DataType::kFLOAT32)
        || !BwdCheckTensor("LayerNorm cpu mean shape/dtype", std::get<1>(fwd_cpu), {1, 2}, DataType::kFLOAT32)
        || !BwdCheckTensor("LayerNorm cpu rstd shape/dtype", std::get<2>(fwd_cpu), {1, 2}, DataType::kFLOAT32)) {
        return;
    }
    auto bwd_cpu
        = infini_train::Dispatcher::Instance()
              .Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
                  {kCPU, "LayerNormBackward"}, x_cpu, w_cpu, b_cpu, std::get<1>(fwd_cpu), std::get<2>(fwd_cpu), g_cpu);
    if (!BwdCheckTensor("LayerNorm cpu dX shape/dtype", std::get<0>(bwd_cpu), input_dims, DataType::kFLOAT32)
        || !BwdCheckTensor("LayerNorm cpu dW shape/dtype", std::get<1>(bwd_cpu), param_dims, DataType::kFLOAT32)
        || !BwdCheckTensor("LayerNorm cpu db shape/dtype", std::get<2>(bwd_cpu), param_dims, DataType::kFLOAT32)) {
        return;
    }
    std::vector<float> cpu_dx = ToHost(std::get<0>(bwd_cpu));
    std::vector<float> cpu_dw = ToHost(std::get<1>(bwd_cpu));
    std::vector<float> cpu_db = ToHost(std::get<2>(bwd_cpu));

    infini_train::core::DeviceGuard guard(Mlu());
    auto x_mlu = MakeTensor(input_dims, input_values, Mlu());
    auto w_mlu = MakeTensor(param_dims, weight_values, Mlu());
    auto b_mlu = MakeTensor(param_dims, bias_values, Mlu());
    auto g_mlu = MakeTensor(input_dims, grad_values, Mlu());
    auto fwd_mlu = infini_train::Dispatcher::Instance()
                       .Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
                           {kMLU, "LayerNormForward"}, x_mlu, w_mlu, b_mlu, eps);
    if (!BwdCheckTensor("LayerNorm fwd out shape/dtype", std::get<0>(fwd_mlu), input_dims, DataType::kFLOAT32)
        || !BwdCheckTensor("LayerNorm mean shape/dtype", std::get<1>(fwd_mlu), {1, 2}, DataType::kFLOAT32)
        || !BwdCheckTensor("LayerNorm rstd shape/dtype", std::get<2>(fwd_mlu), {1, 2}, DataType::kFLOAT32)) {
        return;
    }
    BwdCompare("LayerNorm mean vs cpu", ToDouble(ToHost(std::get<1>(fwd_cpu))), ToHost(std::get<1>(fwd_mlu)), 1e-5);
    BwdCompare("LayerNorm rstd vs cpu", ToDouble(ToHost(std::get<2>(fwd_cpu))), ToHost(std::get<2>(fwd_mlu)), 1e-5);

    auto bwd_mlu
        = infini_train::Dispatcher::Instance()
              .Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
                  {kMLU, "LayerNormBackward"}, x_mlu, w_mlu, b_mlu, std::get<1>(fwd_mlu), std::get<2>(fwd_mlu), g_mlu);
    if (!BwdCheckTensor("LayerNorm dX shape/dtype", std::get<0>(bwd_mlu), input_dims, DataType::kFLOAT32)
        || !BwdCheckTensor("LayerNorm dW shape/dtype", std::get<1>(bwd_mlu), param_dims, DataType::kFLOAT32)
        || !BwdCheckTensor("LayerNorm db shape/dtype", std::get<2>(bwd_mlu), param_dims, DataType::kFLOAT32)) {
        return;
    }
    // 三份梯度各自完成阻塞读回后才记该 case 完成，随后比较全部复用本地 vector
    std::vector<float> dx_host = ToHost(std::get<0>(bwd_mlu));
    std::vector<float> dw_host = ToHost(std::get<1>(bwd_mlu));
    std::vector<float> db_host = ToHost(std::get<2>(bwd_mlu));
    LogBackwardCaseSync("LayerNormBackward");
    BwdCompare("LayerNorm dX vs cpu", ToDouble(cpu_dx), dx_host, 1e-5);
    BwdCompare("LayerNorm dW vs cpu", ToDouble(cpu_dw), dw_host, 1e-5);
    BwdCompare("LayerNorm db vs cpu", ToDouble(cpu_db), db_host, 1e-5);

    // 固定框架测试的第一例期望值（tests/autograd/test_autograd_normalization_backward.cc）
    const std::vector<double> expect_dx{1.60994446, 0.08943637,  -5.00876665, 3.30938578,
                                        0.15652537, -0.02236041, -0.42485276, 0.29068780};
    const std::vector<double> expect_dw{-2.01245522, -0.44721049, 2.23606181, 4.69572210};
    const std::vector<double> expect_db{1.5, 1.0, 5.0, 3.5};
    BwdCompare("LayerNorm dX vs upstream expected", expect_dx, dx_host, 1e-5);
    BwdCompare("LayerNorm dW vs upstream expected", expect_dw, dw_host, 1e-5);
    BwdCompare("LayerNorm db vs upstream expected", expect_db, db_host, 1e-5);
}

void TestMatmulBackward007() {
    struct Case {
        const char *name;
        std::vector<int64_t> a_dims;
        std::vector<int64_t> b_dims;
    };
    const Case cases[]
        = {{"Matmul bwd 2D [4,8]x[8,6]", {4, 8}, {8, 6}}, {"Matmul bwd batch [2,4,8]x[2,8,6]", {2, 4, 8}, {2, 8, 6}}};
    for (const auto &one : cases) {
        const int64_t m = one.a_dims[one.a_dims.size() - 2];
        const int64_t k = one.a_dims[one.a_dims.size() - 1];
        const int64_t n = one.b_dims[one.b_dims.size() - 1];
        int64_t batch = 1;
        for (size_t i = 0; i + 2 < one.a_dims.size(); ++i) { batch *= one.a_dims[i]; }
        const int64_t a_count = batch * m * k;
        const int64_t b_count = batch * k * n;
        const int64_t g_count = batch * m * n;
        std::vector<float> a_host(static_cast<size_t>(a_count));
        std::vector<float> b_host(static_cast<size_t>(b_count));
        std::vector<float> g_host(static_cast<size_t>(g_count));
        for (int64_t i = 0; i < a_count; ++i) {
            a_host[static_cast<size_t>(i)] = static_cast<float>(static_cast<int>(i % 13) - 6) / 32.0f;
        }
        for (int64_t i = 0; i < b_count; ++i) {
            b_host[static_cast<size_t>(i)] = static_cast<float>(static_cast<int>(i % 11) - 5) / 32.0f;
        }
        for (int64_t i = 0; i < g_count; ++i) {
            g_host[static_cast<size_t>(i)] = static_cast<float>(static_cast<int>(i % 7) - 3) / 32.0f;
        }
        std::vector<double> expect_da(static_cast<size_t>(a_count), 0.0);
        std::vector<double> expect_db(static_cast<size_t>(b_count), 0.0);
        for (int64_t bi = 0; bi < batch; ++bi) {
            for (int64_t r = 0; r < m; ++r) {
                for (int64_t c = 0; c < k; ++c) {
                    double acc = 0.0;
                    for (int64_t j = 0; j < n; ++j) {
                        acc += static_cast<double>(g_host[static_cast<size_t>(bi * m * n + r * n + j)])
                             * static_cast<double>(b_host[static_cast<size_t>(bi * k * n + c * n + j)]);
                    }
                    expect_da[static_cast<size_t>(bi * m * k + r * k + c)] = acc;
                }
            }
            for (int64_t c = 0; c < k; ++c) {
                for (int64_t j = 0; j < n; ++j) {
                    double acc = 0.0;
                    for (int64_t r = 0; r < m; ++r) {
                        acc += static_cast<double>(a_host[static_cast<size_t>(bi * m * k + r * k + c)])
                             * static_cast<double>(g_host[static_cast<size_t>(bi * m * n + r * n + j)]);
                    }
                    expect_db[static_cast<size_t>(bi * k * n + c * n + j)] = acc;
                }
            }
        }

        auto a_cpu = MakeTensor(one.a_dims, a_host, Cpu());
        auto b_cpu = MakeTensor(one.b_dims, b_host, Cpu());
        // grad_output 的 shape 是 A_dims[:-1] + B_dims[-1:]（A 末维为 k，需换成 n）
        std::vector<int64_t> g_dims = one.a_dims;
        g_dims.back() = n;
        auto g_cpu = MakeTensor(g_dims, g_host, Cpu());
        auto da_cpu = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kCPU, "MatmulBackwardInput"},
                                                                                         b_cpu, g_cpu, one.a_dims);
        auto db_cpu = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kCPU, "MatmulBackwardOther"},
                                                                                         a_cpu, g_cpu, one.b_dims);
        if (!BwdCheckTensor("Matmul cpu dA shape/dtype", da_cpu, one.a_dims, DataType::kFLOAT32)
            || !BwdCheckTensor("Matmul cpu dB shape/dtype", db_cpu, one.b_dims, DataType::kFLOAT32)) {
            return;
        }

        infini_train::core::DeviceGuard guard(Mlu());
        auto a_mlu = MakeTensor(one.a_dims, a_host, Mlu());
        auto b_mlu = MakeTensor(one.b_dims, b_host, Mlu());
        auto g_mlu = MakeTensor(g_dims, g_host, Mlu());
        auto da_mlu = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kMLU, "MatmulBackwardInput"},
                                                                                         b_mlu, g_mlu, one.a_dims);
        auto db_mlu = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kMLU, "MatmulBackwardOther"},
                                                                                         a_mlu, g_mlu, one.b_dims);
        if (!BwdCheckTensor("Matmul dA shape/dtype", da_mlu, one.a_dims, DataType::kFLOAT32)
            || !BwdCheckTensor("Matmul dB shape/dtype", db_mlu, one.b_dims, DataType::kFLOAT32)) {
            return;
        }
        std::vector<float> da_host = ToHost(da_mlu);
        std::vector<float> db_host = ToHost(db_mlu);
        LogBackwardCaseSync(one.name);
        BwdCompare((std::string(one.name) + " dA vs analytic").c_str(), expect_da, da_host, 1e-6);
        BwdCompare((std::string(one.name) + " dA vs cpu").c_str(), ToDouble(ToHost(da_cpu)), da_host, 1e-6);
        BwdCompare((std::string(one.name) + " dB vs analytic").c_str(), expect_db, db_host, 1e-6);
        BwdCompare((std::string(one.name) + " dB vs cpu").c_str(), ToDouble(ToHost(db_cpu)), db_host, 1e-6);
    }
}

} // namespace

// ---------------------------------------------------------------------------
// 028：BF16 公式级门禁（--bf16-gate）
//
// 目的：在一次 ≤10 s 的设备运行里判定「BF16 这条路能不能走」，四组最小覆盖：
//   1) Cast 双向：fp32->bf16 位模式逐位对框架 host helper；bf16->fp32 精确回读；
//   2) Linear forward/backward 在 BF16 输入下对 double 解析参考（输出本身是 bf16，故真值也圆整）；
//   3) elementwise promote：bf16 + fp32 的结果 dtype 必须是 FP32，且数值对 double 参考；
//   4) BF16 grad + FP32 param 的 AccumulateGrad：**允许显式拒绝，禁止静默错值**
//      （静默错值/静默无操作都算 FAIL，runner 据 SILENT_* 标记分级 S2_GATE_FAILED_PRODUCT_ISSUE）。
//
// 真值来源约束（028/K-001 C5）：一律用框架 host helper
// （infini_train::detail::FloatToBf16Bits / Bf16BitsToFloat）或 double 解析给出，
// 不得用被测 Cast kernel 生成真值，避免循环论证。
// ---------------------------------------------------------------------------
namespace bf16_gate {

std::shared_ptr<Tensor> MakeTensorBf16(const std::vector<int64_t> &dims, const std::vector<float> &host, Device dev) {
    auto t = std::make_shared<Tensor>(dims, DataType::kBFLOAT16, dev);
    std::vector<uint16_t> bits(host.size());
    for (size_t i = 0; i < host.size(); ++i) { bits[i] = infini_train::detail::FloatToBf16Bits(host[i]); }
    if (dev.type() == kCPU) {
        std::memcpy(t->DataPtr(), bits.data(), bits.size() * sizeof(uint16_t));
        return t;
    }
    infini_train::core::GetDeviceGuardImpl(dev.type())
        ->Memcpy(t->DataPtr(), bits.data(), bits.size() * sizeof(uint16_t), infini_train::core::MemcpyKind::kH2D);
    return t;
}

// BF16 只能按 16 bit 读回；按 sizeof(float) 读会错位读取（028/K-001 §2 的缺口）。
std::vector<uint16_t> ToHostBf16Bits(const std::shared_ptr<Tensor> &t) {
    std::vector<uint16_t> bits(t->NumElements());
    if (t->GetDevice().type() == kCPU) {
        std::memcpy(bits.data(), t->DataPtr(), bits.size() * sizeof(uint16_t));
        return bits;
    }
    infini_train::core::GetDeviceGuardImpl(t->GetDevice().type())
        ->Memcpy(bits.data(), t->DataPtr(), bits.size() * sizeof(uint16_t), infini_train::core::MemcpyKind::kD2H);
    return bits;
}

float RoundToBf16(float x) {
    return infini_train::detail::Bf16BitsToFloat(infini_train::detail::FloatToBf16Bits(x));
}

std::vector<float> RoundVec(const std::vector<float> &v) {
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) { out[i] = RoundToBf16(v[i]); }
    return out;
}

// 1) Cast 双向
bool TestCast() {
    const int64_t n = 64;
    std::vector<float> src(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        switch (static_cast<int>(i % 8)) {
        case 0: src[i] = static_cast<float>(i) * 0.0137f; break;
        case 1: src[i] = -static_cast<float>(i) * 0.0211f; break;
        case 2: src[i] = 1.0e-38f * static_cast<float>(i + 1); break; // bf16 次正规附近
        case 3: src[i] = 3.0e38f; break;                              // 大值
        case 4: src[i] = 1.0f + static_cast<float>(i) / 256.0f; break; // 需要舍入
        case 5: src[i] = -1.0f - static_cast<float>(i) / 512.0f; break;
        default: src[i] = static_cast<float>(i % 5) - 2.0f; break;
        }
    }
    std::shared_ptr<Tensor> bf, back;
    {
        infini_train::core::DeviceGuard guard(Mlu());
        auto kernel = infini_train::Dispatcher::Instance().GetKernel({kMLU, "Cast"});
        auto x = MakeTensor({n}, src, Mlu());
        bf = kernel.Call<std::shared_ptr<Tensor>>(x, DataType::kBFLOAT16);
        if (!bf) {
            std::printf("%-46s FAIL null output\n", "bf16 gate cast fp32->bf16");
            return false;
        }
        if (!BwdCheckTensor("bf16 gate cast fp32->bf16", bf, {n}, DataType::kBFLOAT16)) { return false; }
        back = kernel.Call<std::shared_ptr<Tensor>>(bf, DataType::kFLOAT32);
        if (!BwdCheckTensor("bf16 gate cast bf16->fp32", back, {n}, DataType::kFLOAT32)) { return false; }
    }
    const std::vector<uint16_t> bits = ToHostBf16Bits(bf);
    std::vector<float> got_bits(static_cast<size_t>(n));
    std::vector<double> exp_bits(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        got_bits[i] = static_cast<float>(bits[static_cast<size_t>(i)]);
        exp_bits[i] = static_cast<double>(infini_train::detail::FloatToBf16Bits(src[static_cast<size_t>(i)]));
    }
    bool ok = BwdCompare("bf16 gate cast fp32->bf16 bit-exact", exp_bits, got_bits, 0.0);
    const std::vector<float> back_host = ToHost(back);
    std::vector<double> exp_back(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        exp_back[i] = static_cast<double>(infini_train::detail::Bf16BitsToFloat(
            infini_train::detail::FloatToBf16Bits(src[static_cast<size_t>(i)])));
    }
    ok = BwdCompare("bf16 gate cast bf16->fp32 exact", exp_back, back_host, 0.0) && ok;
    LogBackwardCaseSync("bf16 gate cast");
    return ok;
}

// 2) Linear forward/backward（BF16 输入，输出 bf16；真值 = double 解析后再圆整）
bool TestLinear(int64_t rows, int64_t in_f, int64_t out_f) {
    const auto x = bf16_gate::RoundVec(RandVec(static_cast<size_t>(rows * in_f), 1.0f, 401));
    const auto w = bf16_gate::RoundVec(RandVec(static_cast<size_t>(in_f * out_f), 0.05f, 402));
    const auto b = bf16_gate::RoundVec(RandVec(static_cast<size_t>(out_f), 0.02f, 403));
    const auto g = bf16_gate::RoundVec(RandVec(static_cast<size_t>(rows * out_f), 0.5f, 404));
    const std::vector<int64_t> x_dims{rows, in_f};
    const std::vector<int64_t> w_dims{in_f, out_f};
    const std::vector<int64_t> g_dims{rows, out_f};

    std::vector<float> got_y, got_dx, got_dw, got_db;
    {
        infini_train::core::DeviceGuard guard(Mlu());
        auto xt = bf16_gate::MakeTensorBf16(x_dims, x, Mlu());
        auto wt = bf16_gate::MakeTensorBf16(w_dims, w, Mlu());
        auto bt = bf16_gate::MakeTensorBf16({out_f}, b, Mlu());
        auto gt = bf16_gate::MakeTensorBf16(g_dims, g, Mlu());
        auto y = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kMLU, "LinearForward"}, xt, wt,
                                                                                    false, bt);
        if (!BwdCheckTensor("bf16 gate linear fwd", y, {rows, out_f}, DataType::kBFLOAT16)) { return false; }
        auto dx = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
            {kMLU, "LinearBackwardInput"}, wt, gt, false, in_f, out_f, x_dims);
        auto dw = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
            {kMLU, "LinearBackwardWeight"}, xt, gt, false, in_f, out_f);
        auto db = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kMLU, "LinearBackwardBias"}, gt,
                                                                                     out_f);
        if (!dx || !dw || !db) {
            std::printf("%-46s FAIL null backward output\n", "bf16 gate linear bwd");
            return false;
        }
        // 注意：Linear 的 forward/backward 输出 dtype 跟随输入（bf16），必须按 16 bit 读回，
        // 不能用 ToHost（它按 sizeof(float) 拷贝，会把 bf16 缓冲读错/越界）。
        auto read_bf16 = [](const std::shared_ptr<Tensor> &t) {
            const auto bits = ToHostBf16Bits(t);
            std::vector<float> out(bits.size());
            for (size_t i = 0; i < bits.size(); ++i) { out[i] = infini_train::detail::Bf16BitsToFloat(bits[i]); }
            return out;
        };
        got_dx = read_bf16(dx);
        got_dw = read_bf16(dw);
        got_db = read_bf16(db);
        got_y = read_bf16(y);
    }

    std::vector<double> exp_y(static_cast<size_t>(rows * out_f), 0.0);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t o = 0; o < out_f; ++o) {
            double acc = static_cast<double>(b[static_cast<size_t>(o)]);
            for (int64_t i = 0; i < in_f; ++i) {
                acc += static_cast<double>(x[static_cast<size_t>(r * in_f + i)])
                       * static_cast<double>(w[static_cast<size_t>(i * out_f + o)]);
            }
            exp_y[static_cast<size_t>(r * out_f + o)] = static_cast<double>(bf16_gate::RoundToBf16(static_cast<float>(acc)));
        }
    }
    bool ok = BwdCompare("bf16 gate linear fwd vs double(rounded)", exp_y, got_y, 1e-4);

    std::vector<double> exp_dx(static_cast<size_t>(rows * in_f), 0.0);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t i = 0; i < in_f; ++i) {
            double acc = 0.0;
            for (int64_t o = 0; o < out_f; ++o) {
                acc += static_cast<double>(g[static_cast<size_t>(r * out_f + o)])
                       * static_cast<double>(w[static_cast<size_t>(i * out_f + o)]);
            }
            exp_dx[static_cast<size_t>(r * in_f + i)] = static_cast<double>(bf16_gate::RoundToBf16(static_cast<float>(acc)));
        }
    }
    ok = BwdCompare("bf16 gate linear dx vs double(rounded)", exp_dx, got_dx, 1e-4) && ok;

    std::vector<double> exp_dw(static_cast<size_t>(in_f * out_f), 0.0);
    for (int64_t i = 0; i < in_f; ++i) {
        for (int64_t o = 0; o < out_f; ++o) {
            double acc = 0.0;
            for (int64_t r = 0; r < rows; ++r) {
                acc += static_cast<double>(x[static_cast<size_t>(r * in_f + i)])
                       * static_cast<double>(g[static_cast<size_t>(r * out_f + o)]);
            }
            exp_dw[static_cast<size_t>(i * out_f + o)] = static_cast<double>(bf16_gate::RoundToBf16(static_cast<float>(acc)));
        }
    }
    ok = BwdCompare("bf16 gate linear dw vs double(rounded)", exp_dw, got_dw, 1e-4) && ok;

    std::vector<double> exp_db(static_cast<size_t>(out_f), 0.0);
    for (int64_t o = 0; o < out_f; ++o) {
        double acc = 0.0;
        for (int64_t r = 0; r < rows; ++r) { acc += static_cast<double>(g[static_cast<size_t>(r * out_f + o)]); }
        exp_db[static_cast<size_t>(o)] = static_cast<double>(bf16_gate::RoundToBf16(static_cast<float>(acc)));
    }
    ok = BwdCompare("bf16 gate linear db vs double(rounded)", exp_db, got_db, 1e-4) && ok;
    LogBackwardCaseSync("bf16 gate linear");
    return ok;
}

// 3) elementwise promote：bf16 + fp32 的结果 dtype 必须是 FP32
bool TestPromote(int64_t n) {
    const auto a = RandVec(static_cast<size_t>(n), 1.0f, 501);
    const auto ar = bf16_gate::RoundVec(a);
    const auto b = RandVec(static_cast<size_t>(n), 1.0f, 502);
    std::shared_ptr<Tensor> out;
    {
        infini_train::core::DeviceGuard guard(Mlu());
        auto at = bf16_gate::MakeTensorBf16({n}, a, Mlu());
        auto bt = MakeTensor({n}, b, Mlu());
        out = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kMLU, "AddForward"}, at, bt);
    }
    if (!BwdCheckTensor("bf16 gate promote output dtype", out, {n}, DataType::kFLOAT32)) { return false; }
    std::vector<double> expect(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        expect[i] = static_cast<double>(ar[static_cast<size_t>(i)]) + static_cast<double>(b[static_cast<size_t>(i)]);
    }
    const bool ok = BwdCompare("bf16 gate promote bf16+fp32->fp32", expect, ToHost(out), 1e-6);
    LogBackwardCaseSync("bf16 gate promote");
    return ok;
}

// 4) BF16 grad + FP32 param 的 AccumulateGrad
//    返回：0=数值一致（安全），1=静默无操作，2=静默错值（把 fp32 内存按 bf16 解释）
//    说明：若 kernel/框架显式拒绝，进程会 CHECK/异常终止，runner 在日志里看到即按「显式拒绝」处理。
int TestAccumulateGradMismatch() {
    const int64_t n = 64;
    const auto init = RandVec(static_cast<size_t>(n), 1.0f, 601);
    const auto grad = RandVec(static_cast<size_t>(n), 0.5f, 602);
    const auto grad_r = bf16_gate::RoundVec(grad);
    std::vector<float> got;
    {
        infini_train::core::DeviceGuard guard(Mlu());
        auto t = MakeTensor({n}, init, Mlu());              // FP32 target
        auto g = bf16_gate::MakeTensorBf16({n}, grad, Mlu()); // BF16 grad（故意错配）
        auto kernel = infini_train::Dispatcher::Instance().GetKernel({kMLU, "AccumulateGrad"});
        kernel.Call<void>(g, -0.5f, t);
        got = ToHost(t); // 阻塞读回（MluGuardImpl::Memcpy 内部 cnrtQueueSync）
    }
    size_t exact = 0, changed = 0;
    for (int64_t i = 0; i < n; ++i) {
        const double expect = static_cast<double>(init[static_cast<size_t>(i)])
                              - 0.5 * static_cast<double>(grad_r[static_cast<size_t>(i)]);
        const double tol = 1e-6 * std::max(1.0, std::fabs(expect));
        if (std::isfinite(got[static_cast<size_t>(i)]) && std::fabs(static_cast<double>(got[static_cast<size_t>(i)]) - expect) <= tol) {
            ++exact;
        }
        if (got[static_cast<size_t>(i)] != init[static_cast<size_t>(i)]) { ++changed; }
    }
    LogBackwardCaseSync("bf16 gate accumulate_grad mixed dtype");
    if (exact == static_cast<size_t>(n)) {
        std::printf("%-46s EXACT (%zu/%lld)  verdict=VALUE_OK\n", "bf16 gate accumgrad bf16grad+fp32param",
                    exact, static_cast<long long>(n));
        return 0;
    }
    if (changed == 0) {
        std::printf("%-46s UNCHANGED  verdict=SILENT_NOOP (update silently dropped)\n",
                    "bf16 gate accumgrad bf16grad+fp32param");
        return 1;
    }
    std::printf("%-46s MISMATCH changed=%zu/%lld  verdict=SILENT_CORRUPTION (fp32 memory read as bf16)\n",
                "bf16 gate accumgrad bf16grad+fp32param", changed, static_cast<long long>(n));
    return 2;
}

int RunBf16Gate() {
    // 029：CHECK 会 abort，而 stdout 重定向到文件时是全缓冲 —— 不关缓冲就可能在 abort 时丢掉
    // 组 1–3 的 PASS 行，导致"三组仍通过"无法复核。这里显式关闭缓冲。
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("[bf16-gate] begin device=%d\n", g_device);
    int fails = 0;
    if (!TestCast()) { ++fails; }
    if (!TestLinear(3, 5, 7)) { ++fails; }
    if (!TestLinear(8, 16, 16)) { ++fails; }
    if (!TestPromote(64)) { ++fails; }
    // 预声明标记：若产品按约定显式中止（Alt-A），进程会在下面这一行之后终止、不会打印 RESULT，
    // runner 凭"组 1–3 PASS + 本标记 + 固定 mismatch token + 无 RESULT"判定为"明确拒绝"。
    std::printf("[bf16-gate] mixed_dtype_probe begin expect=EXPLICIT_REFUSAL\n");
    std::fflush(stdout);
    const int mixed = TestAccumulateGradMismatch();
    if (mixed != 0) { ++fails; }
    std::printf("[bf16-gate] summary fails=%d mixed_dtype_verdict=%s\n", fails,
                mixed == 0 ? "VALUE_OK" : (mixed == 1 ? "SILENT_NOOP" : "SILENT_CORRUPTION"));
    std::printf("RESULT: %s\n", (fails == 0 && !g_failed) ? "PASS" : "FAIL");
    return (fails == 0 && !g_failed) ? 0 : 1;
}

} // namespace bf16_gate

int main(int argc, char **argv) {
    // 模式解析必须在 backend 注册/设备初始化之前（001-P2 v2 要求）
    std::string probe_case;
    std::string status_test;
    bool comparator_selftest = false;
    bool backward_group_given = false;
    bool bf16_gate = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--comparator-selftest") {
            comparator_selftest = true;
        } else if (arg == "--bf16-gate") {
            bf16_gate = true;
        } else if (arg == "--inject-finite") {
            g_inject_finite = true;
        } else if (arg == "--inject-nan") {
            g_inject_nan = true;
        } else if (arg == "--debug-status-test" && i + 1 < argc) {
            status_test = argv[++i];
        } else if (arg == "--probe-case" && i + 1 < argc) {
            probe_case = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            g_device = std::atoi(argv[++i]);
        } else if (arg == "--backward-group" && i + 1 < argc) {
            g_backward_group = argv[++i];
            backward_group_given = true;
        } else {
            std::printf("unknown arg: %s\n", argv[i]);
            return 2;
        }
    }

    // 007：先区分“是否给了参数”与“取值”，空值同样拒绝（否则会被当成未传而跑默认套件）；
    // 取值与模式冲突也必须在 backend 注册/设备初始化之前拒绝。
    if (backward_group_given
        && (g_backward_group.empty()
            || (g_backward_group != "a" && g_backward_group != "b" && g_backward_group != "all"))) {
        std::printf("invalid --backward-group value: '%s' (expect a|b|all, empty not allowed)\n",
                    g_backward_group.c_str());
        return 2;
    }
    if (backward_group_given && (comparator_selftest || !status_test.empty() || !probe_case.empty())) {
        std::printf(
            "--backward-group cannot be combined with --comparator-selftest/--debug-status-test/--probe-case\n");
        return 2;
    }
    if (bf16_gate
        && (backward_group_given || comparator_selftest || !status_test.empty() || !probe_case.empty())) {
        std::printf("--bf16-gate cannot be combined with --backward-group/--comparator-selftest/"
                    "--debug-status-test/--probe-case\n");
        return 2;
    }

    if (comparator_selftest) {
        return RunComparatorSelftest();
    }

    // S03：受控失败路径测试。只调用既有 CheckSyncResult 的伪返回码，不把任何 queue 交给 CNRT，
    // 也不触发真实设备故障。父脚本按"唯一标识 + 预期终止 + 无后续标记"判定。
    if (!status_test.empty()) {
        // S03: controlled failure-path test. Calls the existing CheckSyncResult with a fabricated
        // return code only; no queue is handed to CNRT and no real device fault is provoked.
        if (status_test == "success") {
            std::cout << "STATUS_TEST_BEGIN success" << std::endl;
            infini_train::core::mlu::CheckSyncResult(cnrtSuccess, "debug-status-test-success");
            std::cout << "STATUS_TEST_END success (continued as expected)" << std::endl;
            return 0;
        }
        if (status_test == "failure") {
            std::cout << "STATUS_TEST_BEGIN failure" << std::endl;
            std::cout << "STATUS_TEST_ABORT_MARKER fake-cnrt-failure-632015" << std::endl;
            std::fflush(stdout);
            infini_train::core::mlu::CheckSyncResult(static_cast<cnrtRet_t>(632015), "debug-status-test-failure");
            std::cout << "STATUS_TEST_AFTER_FATAL should-not-appear" << std::endl;
            return 0;
        }
        std::cout << "unknown --debug-status-test value: " << status_test << std::endl;
        return 2;
    }

    infini_train::backends::mlu::RegisterBackend();
    std::printf("op_test: device=%d inject_finite=%d inject_nan=%d\n", g_device, static_cast<int>(g_inject_finite),
                static_cast<int>(g_inject_nan));

    if (!probe_case.empty()) {
        return RunProbeCase(probe_case);
    }

    // 028：BF16 公式级门禁（独立开关，不影响默认套件与 --backward-group）
    if (bf16_gate) {
        return bf16_gate::RunBf16Gate();
    }

    // 007：关键反向 group（默认套件保持不动，只有显式传参才跑）
    if (!g_backward_group.empty()) {
        const bool run_a = g_backward_group == "a" || g_backward_group == "all";
        const bool run_b = g_backward_group == "b" || g_backward_group == "all";
        std::printf("backward_group=%s sync=blocking_Memcpy_current_queue device=%d\n", g_backward_group.c_str(),
                    g_device);
        if (run_a) {
            TestEmbeddingBackward007();
            TestCrossEntropyBackward007(8);
            TestCrossEntropyBackward007(1);
            TestAccumulateGrad007();
        }
        if (run_b) {
            TestSoftmaxBackward007();
            TestLayerNormBackward007();
            TestMatmulBackward007();
        }
        std::printf("RESULT: %s\n", g_failed ? "FAIL" : "PASS");
        return g_failed ? 1 : 0;
    }

    for (bool transpose : {true, false}) {
        TestLinearForward(3, 5, 7, transpose, true);
        TestLinearForward(3, 5, 7, transpose, false);
        TestLinearBackward(3, 5, 7, transpose);
    }
    TestPowAnalytic();
    TestCrossEntropy(4, 8);
    TestLayerNorm(8, 16);
    TestSoftmax(8, 16);
    TestAddBroadcast(8, 16);
    TestMulScalar(64, 0.5f);
    TestTranspose(1, 2, 4, 8);

    std::printf("RESULT: %s\n", g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0; // 注入模式下同样如实返回非 0，由外层脚本检查预期退出码
}
