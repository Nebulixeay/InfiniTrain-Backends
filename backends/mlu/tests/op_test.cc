// Operator comparisons against CPU and independent analytic references.
// Non-finite references require explicit opt-in and matching NaN/Inf classification.
// Finite references require finite results within atol + rtol * abs(reference).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
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

namespace {

using infini_train::DataType;
using infini_train::Device;
using infini_train::Tensor;

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

// Classify non-finite values before applying tolerances to finite pairs.
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
            // Non-finite references require opt-in and matching classification/sign.
            const bool same = (a_nan && b_nan) || (a_inf && b_inf && ((a > 0) == (b > 0)));
            if (!allow_special_ref || !same) {
                ++v.bad;
                v.worst = i;
                v.max_abs = std::numeric_limits<double>::infinity();
            }
            continue;
        }
        if (b_nan || b_inf) { // Finite reference requires a finite result.
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

// Check independent analytic expectations, without relying on the CPU kernel.
bool CompareAnalytic(const std::string &name, const std::vector<double> &expect, const std::vector<float> &got,
                     double atol, double rtol) {
    std::vector<float> ref(expect.size());
    for (size_t i = 0; i < expect.size(); ++i) { ref[i] = static_cast<float>(expect[i]); }
    return Compare(name, ref, got, atol, rtol, /*allow_special_ref=*/true);
}

// Host-only comparator self-test; no device initialization.
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
        {2.0f, 3.0f, true, 9.0, "base=3 exp=2 -> 9"}, // tensor is the exponent; scalar is the base.
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
        // Check analytic values first, then compare against CPU.
        CompareAnalytic(std::string(name) + " [analytic]", {c.expect}, got, 1e-6, 1e-5);
        Compare(std::string(name) + " [vs cpu]", cpu, got, 1e-6, 1e-5, /*allow_special_ref=*/true);
    }

    // For y = base^x, dy/dx = base^x * log(base).
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

    // Include zero and negative inputs for both forward and backward.
    {
        const std::vector<float> xs = {0.0f, -2.0f, 0.5f, 3.0f};
        const std::vector<float> gs = {1.0f, 2.0f, -1.0f, 0.5f};
        std::vector<double> exp_f(4), exp_b(4);
        for (size_t i = 0; i < xs.size(); ++i) {
            const double x = xs[i];
            exp_f[i] = x * x * x;
            exp_b[i] = static_cast<double>(gs[i]) * 3.0 * x * x; // Zero at x = 0.
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

void TestDenseBinaryForward() {
    constexpr int64_t side = 257;
    auto left = RandVec(side * side, 1.0f, 71);
    auto right = RandVec(side * side, 1.0f, 72);
    auto a = MakeTensor({side, side}, left, Mlu());
    auto b = MakeTensor({side, side}, right, Mlu());
    for (const std::string op : {"AddForward", "MulForward"}) {
        std::vector<float> expected(left.size());
        for (size_t i = 0; i < left.size(); ++i) {
            expected[i] = op == "AddForward" ? left[i] + right[i] : left[i] * right[i];
        }
        auto output = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kMLU, op}, a, b);
        Compare(op + " dense FP32 257x257", expected, ToHost(output), 0.0, 0.0);
    }
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

// Backward regressions use blocking ToHost copies on the current device stream.
// A DeviceGuard keeps kernel execution and copy-back on the same device/queue.
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

// Call inside the test DeviceGuard so the recorded stream matches execution.
std::string MluStreamTag() {
    auto *impl = infini_train::core::GetDeviceGuardImpl(Device::DeviceType::kPrivateUse1);
    auto *stream = impl->GetStream(Mlu());
    char buf[128];
    std::snprintf(buf, sizeof(buf), "device=%d stream=%p", g_device, static_cast<void *>(stream));
    return std::string(buf);
}

// Log completion only after blocking ToHost returns; CNRT errors abort inside Memcpy.
void LogBackwardCaseSync(const char *name) {
    std::printf("[bwd] case=%s %s sync=blocking_Memcpy_current_queue host_read_after_tohost_returned "
                "internal_check=MLU_CNRT_CHECK\n",
                name, MluStreamTag().c_str());
}

// Validate size and finiteness before absolute-tolerance comparison. Inject faults only into the result copy.
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
        if (!std::isfinite(ref[i])) { // double-to-float overflow
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

void TestEmbeddingBackward() {
    const std::vector<int64_t> indices{0, 3, 3, 7, 0, 7, 7, 1};
    const int64_t tokens = static_cast<int64_t>(indices.size());
    const int64_t vocab = 32;
    const int64_t dim = 16;
    const std::vector<int64_t> weight_dims{vocab, dim};
    std::vector<float> grad(static_cast<size_t>(tokens * dim));
    for (size_t i = 0; i < grad.size(); ++i) { grad[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 64.0f; }
    // Independent double scatter-add reference, including repeated token IDs.
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

void TestCrossEntropyBackward(int64_t rows) {
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

void TestAccumulateGrad() {
    const int64_t n = 64;
    std::vector<float> init(static_cast<size_t>(n));
    std::vector<float> grad(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        init[static_cast<size_t>(i)] = static_cast<float>(static_cast<int>(i % 9) - 4) / 8.0f;
        if (init[static_cast<size_t>(i)] == 0.0f) { // Require nonzero initial values.
            init[static_cast<size_t>(i)] = 0.125f;
        }
        grad[static_cast<size_t>(i)] = static_cast<float>(static_cast<int>(i % 7) - 3) / 16.0f;
        if (grad[static_cast<size_t>(i)] == 0.0f) { // Require nonzero gradients.
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
            // Revalidate the in-place tensor before copying back the result.
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
            // Revalidate the in-place tensor before copying back the result.
            if (!BwdCheckTensor("AccumulateGrad cpu target after call", t_cpu, {n}, DataType::kFLOAT32)) {
                return;
            }
            ++assertions;
            if (!BwdCompare((std::string(role.name) + " vs cpu").c_str(), ToDouble(ToHost(t_cpu)), mlu_host, 1e-6)) {
                ++failures;
            }
        }
    }
    std::printf("%-46s neutral: assertions=%d failed=%d (kernel-level rate, not the training accumulation rate)\n",
                "AccumulateGrad roles finished", assertions, failures);
}

void TestSoftmaxBackward() {
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
        // Independent stable softmax and analytic backward reference in double.
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
        if (!y_cpu_finite) { // Reject non-finite CPU forward output before backward.
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
                for (int64_t c = 1; c < cols; c += 2) { // Mask odd columns.
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
                for (int64_t c = 1; c < cols; c += 2) { // Mask odd columns.
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

void TestLayerNormBackward() {
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
    // Compare all three gradients only after their blocking copies complete.
    std::vector<float> dx_host = ToHost(std::get<0>(bwd_mlu));
    std::vector<float> dw_host = ToHost(std::get<1>(bwd_mlu));
    std::vector<float> db_host = ToHost(std::get<2>(bwd_mlu));
    LogBackwardCaseSync("LayerNormBackward");
    BwdCompare("LayerNorm dX vs cpu", ToDouble(cpu_dx), dx_host, 1e-5);
    BwdCompare("LayerNorm dW vs cpu", ToDouble(cpu_dw), dw_host, 1e-5);
    BwdCompare("LayerNorm db vs cpu", ToDouble(cpu_db), db_host, 1e-5);

    // Reference the first case in tests/autograd/test_autograd_normalization_backward.cc.
    const std::vector<double> expect_dx{1.60994446, 0.08943637,  -5.00876665, 3.30938578,
                                        0.15652537, -0.02236041, -0.42485276, 0.29068780};
    const std::vector<double> expect_dw{-2.01245522, -0.44721049, 2.23606181, 4.69572210};
    const std::vector<double> expect_db{1.5, 1.0, 5.0, 3.5};
    BwdCompare("LayerNorm dX vs upstream expected", expect_dx, dx_host, 1e-5);
    BwdCompare("LayerNorm dW vs upstream expected", expect_dw, dw_host, 1e-5);
    BwdCompare("LayerNorm db vs upstream expected", expect_db, db_host, 1e-5);
}

void TestMatmulBackward() {
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
        // The output shape replaces the last input dimension k with n.
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

// BF16 tests use host conversion helpers and analytic references, not the cast kernel under test.
// Mixed-dtype accumulation must reject explicitly rather than corrupting or skipping updates.
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

// Copy BF16 storage as 16-bit elements, never as float-sized elements.
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

float RoundToBf16(float x) { return infini_train::detail::Bf16BitsToFloat(infini_train::detail::FloatToBf16Bits(x)); }

std::vector<float> RoundVec(const std::vector<float> &v) {
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) { out[i] = RoundToBf16(v[i]); }
    return out;
}

bool TestCast(int64_t n = 64) {
    std::printf("[cast] elements=%lld\n", static_cast<long long>(n));
    std::vector<float> src(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        switch (static_cast<int>(i % 8)) {
        case 0:
            src[i] = static_cast<float>(i) * 0.0137f;
            break;
        case 1:
            src[i] = -static_cast<float>(i) * 0.0211f;
            break;
        case 2:
            src[i] = 1.0e-38f * static_cast<float>(i + 1);
            break; // Near BF16 subnormals.
        case 3:
            src[i] = 3.0e38f;
            break; // Large values.
        case 4:
            src[i] = 1.0f + static_cast<float>(i) / 256.0f;
            break; // Requires rounding.
        case 5:
            src[i] = -1.0f - static_cast<float>(i) / 512.0f;
            break;
        default:
            src[i] = static_cast<float>(i % 5) - 2.0f;
            break;
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
        if (!BwdCheckTensor("bf16 gate cast fp32->bf16", bf, {n}, DataType::kBFLOAT16)) {
            return false;
        }
        back = kernel.Call<std::shared_ptr<Tensor>>(bf, DataType::kFLOAT32);
        if (!BwdCheckTensor("bf16 gate cast bf16->fp32", back, {n}, DataType::kFLOAT32)) {
            return false;
        }
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
        exp_back[i] = static_cast<double>(
            infini_train::detail::Bf16BitsToFloat(infini_train::detail::FloatToBf16Bits(src[static_cast<size_t>(i)])));
    }
    ok = BwdCompare("bf16 gate cast bf16->fp32 exact", exp_back, back_host, 0.0) && ok;
    LogBackwardCaseSync("bf16 gate cast");
    return ok;
}

bool TestCastBitPatterns() {
    constexpr int64_t count = 65536;
    const uint32_t low_bits[] = {0, 1, 0x7fff, 0x8000, 0x8001, 0xffff};
    std::vector<float> source(count * 6);
    for (int64_t hi = 0; hi < count; ++hi) {
        for (int64_t lo = 0; lo < 6; ++lo) {
            const uint32_t bits = (static_cast<uint32_t>(hi) << 16) | low_bits[lo];
            std::memcpy(&source[hi * 6 + lo], &bits, sizeof(bits));
        }
    }
    infini_train::core::DeviceGuard guard(Mlu());
    auto kernel = infini_train::Dispatcher::Instance().GetKernel({kMLU, "Cast"});
    auto input = MakeTensor({count * 6}, source, Mlu());
    const auto reduced = kernel.Call<std::shared_ptr<Tensor>>(input, DataType::kBFLOAT16);
    const auto result = ToHostBf16Bits(reduced);
    size_t bad = 0;
    for (size_t i = 0; i < source.size(); ++i) {
        if (std::isnan(source[i])) {
            bad += (result[i] & 0x7f80u) != 0x7f80u || (result[i] & 0x7fu) == 0;
        } else {
            bad += result[i] != infini_train::detail::FloatToBf16Bits(source[i]);
        }
    }
    // Construct every BF16 representation directly, independent of the FP32->BF16 kernel.
    std::vector<uint16_t> bits(count);
    for (int64_t i = 0; i < count; ++i) { bits[i] = static_cast<uint16_t>(i); }
    auto bf = std::make_shared<Tensor>(std::vector<int64_t>{count}, DataType::kBFLOAT16, Mlu());
    infini_train::core::GetDeviceGuardImpl(kMLU)->Memcpy(bf->DataPtr(), bits.data(), bits.size() * sizeof(uint16_t),
                                                         infini_train::core::MemcpyKind::kH2D);
    const auto expanded = ToHost(kernel.Call<std::shared_ptr<Tensor>>(bf, DataType::kFLOAT32));
    for (size_t i = 0; i < bits.size(); ++i) {
        if ((bits[i] & 0x7f80u) == 0x7f80u && (bits[i] & 0x7fu) != 0) {
            bad += !std::isnan(expanded[i]);
        } else {
            uint32_t actual = 0;
            std::memcpy(&actual, &expanded[i], sizeof(actual));
            bad += actual != (static_cast<uint32_t>(bits[i]) << 16);
        }
    }
    std::printf("Cast rounding/zeros/subnormals/Inf/NaN: %s bad=%zu\n", bad == 0 ? "PASS" : "FAIL", bad);
    return bad == 0;
}

int RunCastGate() {
    bool ok = TestCastBitPatterns();
    // Cover scalar fallback, tile edges, all launch tasks, and GPT-2 activation/weight sizes.
    for (const int64_t n : {1, 64, 8191, 8192, 8193, 131071, 131072, 131073, 196608, 2359296}) {
        ok = TestCast(n) && ok;
    }
    std::printf("RESULT: %s\n", ok && !g_failed ? "PASS" : "FAIL");
    return ok && !g_failed ? 0 : 1;
}

// Linear BF16 forward output stays BF16; backward outputs match MACA FP32 storage.
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
        if (!BwdCheckTensor("bf16 gate linear fwd", y, {rows, out_f}, DataType::kBFLOAT16)) {
            return false;
        }
        auto dx = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kMLU, "LinearBackwardInput"}, wt,
                                                                                     gt, false, in_f, out_f, x_dims);
        auto dw = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kMLU, "LinearBackwardWeight"}, xt,
                                                                                     gt, false, in_f, out_f);
        auto db = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kMLU, "LinearBackwardBias"}, gt,
                                                                                     out_f);
        if (!dx || !dw || !db) {
            std::printf("%-46s FAIL null backward output\n", "bf16 gate linear bwd");
            return false;
        }
        if (!BwdCheckTensor("bf16 linear dx dtype", dx, x_dims, DataType::kFLOAT32)
            || !BwdCheckTensor("bf16 linear dw dtype", dw, w_dims, DataType::kFLOAT32)
            || !BwdCheckTensor("bf16 linear db dtype", db, {out_f}, DataType::kFLOAT32)) {
            return false;
        }
        // Only the forward result uses BF16 storage.
        auto read_bf16 = [](const std::shared_ptr<Tensor> &t) {
            const auto bits = ToHostBf16Bits(t);
            std::vector<float> out(bits.size());
            for (size_t i = 0; i < bits.size(); ++i) { out[i] = infini_train::detail::Bf16BitsToFloat(bits[i]); }
            return out;
        };
        got_dx = ToHost(dx);
        got_dw = ToHost(dw);
        got_db = ToHost(db);
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
            exp_y[static_cast<size_t>(r * out_f + o)]
                = static_cast<double>(bf16_gate::RoundToBf16(static_cast<float>(acc)));
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
            exp_dx[static_cast<size_t>(r * in_f + i)] = acc;
        }
    }
    ok = BwdCompare("bf16 gate linear dx vs double", exp_dx, got_dx, 1e-5) && ok;

    std::vector<double> exp_dw(static_cast<size_t>(in_f * out_f), 0.0);
    for (int64_t i = 0; i < in_f; ++i) {
        for (int64_t o = 0; o < out_f; ++o) {
            double acc = 0.0;
            for (int64_t r = 0; r < rows; ++r) {
                acc += static_cast<double>(x[static_cast<size_t>(r * in_f + i)])
                     * static_cast<double>(g[static_cast<size_t>(r * out_f + o)]);
            }
            exp_dw[static_cast<size_t>(i * out_f + o)] = acc;
        }
    }
    ok = BwdCompare("bf16 gate linear dw vs double", exp_dw, got_dw, 1e-5) && ok;

    std::vector<double> exp_db(static_cast<size_t>(out_f), 0.0);
    for (int64_t o = 0; o < out_f; ++o) {
        double acc = 0.0;
        for (int64_t r = 0; r < rows; ++r) { acc += static_cast<double>(g[static_cast<size_t>(r * out_f + o)]); }
        exp_db[static_cast<size_t>(o)] = acc;
    }
    ok = BwdCompare("bf16 gate linear db vs double", exp_db, got_db, 1e-5) && ok;
    LogBackwardCaseSync("bf16 gate linear");
    return ok;
}

// Independent double formulas cover mixed dtypes, both Linear layouts and batched Matmul.
// Inputs use non-BF16-exact values to detect missing input rounding as well as output rounding.
bool TestBackwardDtypes(DataType compute, DataType incoming, int64_t batches, bool transpose) {
    const int64_t m = 3, k = 5, n = 7;
    auto a = RandVec(batches * m * k, 0.7f, 701);
    auto b = RandVec(batches * k * n, 0.6f, 702);
    auto g = RandVec(batches * m * n, 0.8f, 703);
    auto tensor = [](const std::vector<int64_t> &dims, const std::vector<float> &v, DataType dt) {
        return dt == DataType::kBFLOAT16 ? MakeTensorBf16(dims, v, Mlu()) : MakeTensor(dims, v, Mlu());
    };
    std::vector<int64_t> ad{m, k}, bd{transpose ? n : k, transpose ? k : n}, gd{m, n};
    if (batches > 1) {
        ad.insert(ad.begin(), batches);
        bd.insert(bd.begin(), batches);
        gd.insert(gd.begin(), batches);
    }
    infini_train::core::DeviceGuard guard(Mlu());
    auto at = tensor(ad, a, compute), bt = tensor(bd, b, compute), gt = tensor(gd, g, incoming);
    if (compute == DataType::kBFLOAT16) {
        a = RoundVec(a);
        b = RoundVec(b);
    }
    if (incoming == DataType::kBFLOAT16) {
        g = RoundVec(g);
    }
    const auto bias_g = g;
    if (compute == DataType::kBFLOAT16) {
        g = RoundVec(g);
    }
    std::vector<double> da(a.size(), 0.0), db(b.size(), 0.0), bias(n, 0.0);
    for (int64_t batch = 0; batch < batches; ++batch) {
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t out = 0; out < n; ++out) {
                const auto gi = batch * m * n + row * n + out;
                bias[out] += static_cast<double>(bias_g[gi]);
                for (int64_t in = 0; in < k; ++in) {
                    const auto ai = batch * m * k + row * k + in;
                    const auto bi = batch * k * n + (transpose ? out * k + in : in * n + out);
                    da[ai] += static_cast<double>(g[gi]) * b[bi];
                    db[bi] += static_cast<double>(a[ai]) * g[gi];
                }
            }
        }
    }
    std::printf("backward_dtype_case compute=%s incoming=%s batches=%lld transpose=%d\n",
                compute == DataType::kBFLOAT16 ? "bf16" : "fp32", incoming == DataType::kBFLOAT16 ? "bf16" : "fp32",
                static_cast<long long>(batches), transpose);
    bool ok = true;
    auto check = [&](const char *name, const std::shared_ptr<Tensor> &out, const std::vector<int64_t> &dims,
                     const std::vector<double> &expected) {
        const bool valid = BwdCheckTensor(name, out, dims, DataType::kFLOAT32);
        ok = valid && ok;
        if (valid) {
            ok = BwdCompare(name, expected, ToHost(out), 1e-5) && ok;
        }
    };
    if (batches == 1) {
        check("LinearBackwardInput dtype+numeric",
              infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kMLU, "LinearBackwardInput"}, bt, gt,
                                                                                 transpose, k, n, ad),
              ad, da);
        check("LinearBackwardWeight dtype+numeric",
              infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kMLU, "LinearBackwardWeight"}, at, gt,
                                                                                 transpose, k, n),
              bd, db);
        check("LinearBackwardBias dtype+numeric",
              infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kMLU, "LinearBackwardBias"}, gt, n),
              {n}, bias);
    }
    if (!transpose) {
        check("MatmulBackwardInput dtype+numeric",
              infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kMLU, "MatmulBackwardInput"}, bt, gt,
                                                                                 ad),
              ad, da);
        check("MatmulBackwardOther dtype+numeric",
              infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kMLU, "MatmulBackwardOther"}, at, gt,
                                                                                 bd),
              bd, db);
    }
    return ok;
}

// Independent analytic gradient with explicitly rounded stored inputs.
bool TestSoftmaxBackwardDtypes(DataType saved, DataType incoming, const std::vector<int64_t> &dims, int64_t dim) {
    const int64_t axis = dim < 0 ? dim + static_cast<int64_t>(dims.size()) : dim;
    int64_t outer = 1, inner = 1, width = dims[axis];
    for (int64_t i = 0; i < axis; ++i) { outer *= dims[i]; }
    for (size_t i = axis + 1; i < dims.size(); ++i) { inner *= dims[i]; }
    const int64_t count = outer * width * inner;
    auto y = RandVec(count, 1.0f, 801);
    auto g = RandVec(count, 0.7f, 802);
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            double sum = 0;
            for (int64_t a = 0; a < width; ++a) {
                auto index = (o * width + a) * inner + i;
                // Include exact masked zeros without making an all-zero row.
                y[index] = a == width - 1 ? 0.0f : std::exp(y[index]);
                sum += y[index];
            }
            for (int64_t a = 0; a < width; ++a) {
                auto index = (o * width + a) * inner + i;
                y[index] = static_cast<float>(y[index] / sum);
            }
        }
    }
    if (saved == DataType::kBFLOAT16) {
        y = RoundVec(y);
    }
    if (incoming == DataType::kBFLOAT16) {
        g = RoundVec(g);
    }
    const auto expected_dtype
        = saved == DataType::kBFLOAT16 && incoming == DataType::kBFLOAT16 ? DataType::kBFLOAT16 : DataType::kFLOAT32;
    std::vector<double> expected(count);
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            double dot = 0;
            for (int64_t a = 0; a < width; ++a) {
                auto index = (o * width + a) * inner + i;
                dot += static_cast<double>(y[index]) * g[index];
            }
            for (int64_t a = 0; a < width; ++a) {
                auto index = (o * width + a) * inner + i;
                double value = static_cast<double>(y[index]) * (g[index] - dot);
                expected[index]
                    = expected_dtype == DataType::kBFLOAT16 ? RoundToBf16(static_cast<float>(value)) : value;
            }
        }
    }
    infini_train::core::DeviceGuard guard(Mlu());
    auto yt = saved == DataType::kBFLOAT16 ? MakeTensorBf16(dims, y, Mlu()) : MakeTensor(dims, y, Mlu());
    auto gt = incoming == DataType::kBFLOAT16 ? MakeTensorBf16(dims, g, Mlu()) : MakeTensor(dims, g, Mlu());
    auto out
        = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({kMLU, "SoftmaxBackward"}, gt, yt, dim);
    std::printf("softmax_backward saved=%s incoming=%s outer=%lld axis=%lld inner=%lld dim=%lld\n",
                saved == DataType::kBFLOAT16 ? "bf16" : "fp32", incoming == DataType::kBFLOAT16 ? "bf16" : "fp32",
                static_cast<long long>(outer), static_cast<long long>(width), static_cast<long long>(inner),
                static_cast<long long>(dim));
    const bool valid = BwdCheckTensor("SoftmaxBackward dtype+shape", out, dims, expected_dtype);
    if (!out || out->Dims() != dims) {
        return false;
    }
    std::vector<float> got;
    if (out->Dtype() == DataType::kBFLOAT16) {
        for (auto bits : ToHostBf16Bits(out)) { got.push_back(infini_train::detail::Bf16BitsToFloat(bits)); }
    } else if (out->Dtype() == DataType::kFLOAT32) {
        got = ToHost(out);
    } else {
        return false;
    }
    LogBackwardCaseSync("SoftmaxBackward mixed dtypes");
    return BwdCompare("SoftmaxBackward analytic", expected, got, 1e-5) && valid;
}

int RunSoftmaxBackwardGate() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    bool ok = true;
    for (auto saved : {DataType::kFLOAT32, DataType::kBFLOAT16}) {
        for (auto incoming : {DataType::kFLOAT32, DataType::kBFLOAT16}) {
            ok = TestSoftmaxBackwardDtypes(saved, incoming, {2, 7}, -1) && ok;
            ok = TestSoftmaxBackwardDtypes(saved, incoming, {2, 7, 3}, 1) && ok;
        }
    }
    ok = TestSoftmaxBackwardDtypes(DataType::kBFLOAT16, DataType::kFLOAT32, {4, 12, 64, 64}, -1) && ok;
    std::printf("RESULT: %s\n", ok && !g_failed ? "PASS" : "FAIL");
    return ok && !g_failed ? 0 : 1;
}

int RunLinearBackwardGate() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    bool ok = TestLinear(3, 5, 7); // Forward remains BF16, backward is FP32.
    for (auto compute : {DataType::kFLOAT32, DataType::kBFLOAT16}) {
        for (auto incoming : {DataType::kFLOAT32, DataType::kBFLOAT16}) {
            ok = TestBackwardDtypes(compute, incoming, 1, false) && ok;
            ok = TestBackwardDtypes(compute, incoming, 1, true) && ok;
            ok = TestBackwardDtypes(compute, incoming, 2, false) && ok;
        }
    }
    std::printf("LINEAR_BACKWARD_GATE: %s\n", ok && !g_failed ? "PASS" : "FAIL");
    return ok && !g_failed ? 0 : 1;
}

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
    if (!BwdCheckTensor("bf16 gate promote output dtype", out, {n}, DataType::kFLOAT32)) {
        return false;
    }
    std::vector<double> expect(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        expect[i] = static_cast<double>(ar[static_cast<size_t>(i)]) + static_cast<double>(b[static_cast<size_t>(i)]);
    }
    const bool ok = BwdCompare("bf16 gate promote bf16+fp32->fp32", expect, ToHost(out), 1e-6);
    LogBackwardCaseSync("bf16 gate promote");
    return ok;
}

// Return 0 for correct values, 1 for a silent no-op, or 2 for corruption.
// An explicit product rejection aborts and is checked by the caller.
int TestAccumulateGradMismatch() {
    const int64_t n = 64;
    const auto init = RandVec(static_cast<size_t>(n), 1.0f, 601);
    const auto grad = RandVec(static_cast<size_t>(n), 0.5f, 602);
    const auto grad_r = bf16_gate::RoundVec(grad);
    std::vector<float> got;
    {
        infini_train::core::DeviceGuard guard(Mlu());
        auto t = MakeTensor({n}, init, Mlu());                // FP32 target
        auto g = bf16_gate::MakeTensorBf16({n}, grad, Mlu()); // Intentionally mismatched BF16 gradient.
        auto kernel = infini_train::Dispatcher::Instance().GetKernel({kMLU, "AccumulateGrad"});
        kernel.Call<void>(g, -0.5f, t);
        got = ToHost(t); // Blocking copy via MluGuardImpl::Memcpy.
    }
    size_t exact = 0, changed = 0;
    for (int64_t i = 0; i < n; ++i) {
        const double expect = static_cast<double>(init[static_cast<size_t>(i)])
                            - 0.5 * static_cast<double>(grad_r[static_cast<size_t>(i)]);
        const double tol = 1e-6 * std::max(1.0, std::fabs(expect));
        if (std::isfinite(got[static_cast<size_t>(i)])
            && std::fabs(static_cast<double>(got[static_cast<size_t>(i)]) - expect) <= tol) {
            ++exact;
        }
        if (got[static_cast<size_t>(i)] != init[static_cast<size_t>(i)]) {
            ++changed;
        }
    }
    LogBackwardCaseSync("bf16 gate accumulate_grad mixed dtype");
    if (exact == static_cast<size_t>(n)) {
        std::printf("%-46s EXACT (%zu/%lld)  verdict=VALUE_OK\n", "bf16 gate accumgrad bf16grad+fp32param", exact,
                    static_cast<long long>(n));
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
    // Disable stdout buffering so preceding test results survive an expected rejection.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("[bf16-gate] begin device=%d\n", g_device);
    int fails = 0;
    if (!TestCast()) {
        ++fails;
    }
    if (!TestLinear(3, 5, 7)) {
        ++fails;
    }
    if (!TestLinear(8, 16, 16)) {
        ++fails;
    }
    if (!TestPromote(64)) {
        ++fails;
    }
    // An expected dtype rejection terminates after this marker without printing RESULT.
    std::printf("[bf16-gate] mixed_dtype_probe begin expect=EXPLICIT_REFUSAL\n");
    std::fflush(stdout);
    const int mixed = TestAccumulateGradMismatch();
    if (mixed != 0) {
        ++fails;
    }
    std::printf("[bf16-gate] summary fails=%d mixed_dtype_verdict=%s\n", fails,
                mixed == 0 ? "VALUE_OK" : (mixed == 1 ? "SILENT_NOOP" : "SILENT_CORRUPTION"));
    std::printf("RESULT: %s\n", (fails == 0 && !g_failed) ? "PASS" : "FAIL");
    return (fails == 0 && !g_failed) ? 0 : 1;
}

} // namespace bf16_gate

// CPU tensors suffice: invalid inputs must be rejected before accessing an MLU queue.
int RunInvalidInputCase(const std::string &name) {
    auto &dispatcher = infini_train::Dispatcher::Instance();
    if (name == "slice-zero-step") {
        auto input = MakeTensor({2}, {1.0f, 2.0f}, Cpu());
        dispatcher.GetKernel({kMLU, "SliceForward"})
            .Call<std::shared_ptr<Tensor>>(input, std::vector<int64_t>{0}, std::vector<int64_t>{2},
                                           std::vector<int64_t>{0});
    } else if (name == "layernorm-mixed-dtype") {
        auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2}, DataType::kBFLOAT16, Cpu());
        auto weight = MakeTensor({2}, {1.0f, 1.0f}, Cpu());
        auto bias = MakeTensor({2}, {0.0f, 0.0f}, Cpu());
        using Outputs = std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>;
        dispatcher.GetKernel({kMLU, "LayerNormForward"}).Call<Outputs>(input, weight, bias, 1e-5f);
    } else {
        return 2;
    }
    std::printf("FAIL: invalid input was accepted\n");
    return 1;
}

int main(int argc, char **argv) {
    // Validate arguments before backend registration or device initialization.
    bool comparator_selftest = false;
    bool backward_group_given = false;
    bool bf16_gate = false;
    bool cast_gate = false;
    bool linear_backward_gate = false;
    bool softmax_backward_gate = false;
    std::string invalid_input_case;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--comparator-selftest") {
            comparator_selftest = true;
        } else if (arg == "--invalid-input-case" && i + 1 < argc) {
            invalid_input_case = argv[++i];
            if (invalid_input_case != "slice-zero-step" && invalid_input_case != "layernorm-mixed-dtype") {
                return 2;
            }
        } else if (arg == "--softmax-backward-gate") {
            softmax_backward_gate = true;
        } else if (arg == "--linear-backward-gate") {
            linear_backward_gate = true;
        } else if (arg == "--cast-gate") {
            cast_gate = true;
        } else if (arg == "--bf16-gate") {
            bf16_gate = true;
        } else if (arg == "--inject-finite") {
            g_inject_finite = true;
        } else if (arg == "--inject-nan") {
            g_inject_nan = true;
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

    // Reject empty/invalid group values before device initialization.
    if (backward_group_given
        && (g_backward_group.empty()
            || (g_backward_group != "a" && g_backward_group != "b" && g_backward_group != "all"))) {
        std::printf("invalid --backward-group value: '%s' (expect a|b|all, empty not allowed)\n",
                    g_backward_group.c_str());
        return 2;
    }
    if (static_cast<int>(backward_group_given) + static_cast<int>(bf16_gate) + static_cast<int>(linear_backward_gate)
            + static_cast<int>(softmax_backward_gate) + static_cast<int>(comparator_selftest)
            + static_cast<int>(!invalid_input_case.empty()) + static_cast<int>(cast_gate)
        > 1) {
        std::printf("select only one test suite\n");
        return 2;
    }
    if (comparator_selftest) {
        return RunComparatorSelftest();
    }
    if (!invalid_input_case.empty()) {
        return RunInvalidInputCase(invalid_input_case);
    }

    infini_train::backends::mlu::RegisterBackend();
    std::printf("op_test: device=%d inject_finite=%d inject_nan=%d\n", g_device, static_cast<int>(g_inject_finite),
                static_cast<int>(g_inject_nan));

    if (cast_gate) {
        return bf16_gate::RunCastGate();
    }
    if (softmax_backward_gate) {
        return bf16_gate::RunSoftmaxBackwardGate();
    }
    if (linear_backward_gate) {
        return bf16_gate::RunLinearBackwardGate();
    }

    // BF16 tests use host conversion helpers and analytic references, not the cast kernel under test.
    // Mixed-dtype accumulation must reject explicitly rather than corrupting or skipping updates.
    if (bf16_gate) {
        return bf16_gate::RunBf16Gate();
    }

    // Run the backward suite only when explicitly requested.
    if (!g_backward_group.empty()) {
        const bool run_a = g_backward_group == "a" || g_backward_group == "all";
        const bool run_b = g_backward_group == "b" || g_backward_group == "all";
        std::printf("backward_group=%s sync=blocking_Memcpy_current_queue device=%d\n", g_backward_group.c_str(),
                    g_device);
        if (run_a) {
            TestEmbeddingBackward();
            TestCrossEntropyBackward(8);
            TestCrossEntropyBackward(1);
            TestAccumulateGrad();
        }
        if (run_b) {
            TestSoftmaxBackward();
            TestLayerNormBackward();
            TestMatmulBackward();
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
    TestDenseBinaryForward();
    TestMulScalar(64, 0.5f);
    TestTranspose(1, 2, 4, 8);

    std::printf("RESULT: %s\n", g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0; // Injected faults must also return a failure.
}
