#include "runtime/mlu_guard_impl.h"

#include <array>
#include <memory>
#include <mutex>
#include <ostream>
#include <vector>

#include "glog/logging.h"
#ifdef USE_CNNL
#include <cnnl.h>
#endif
#include <cnrt.h>

#include "infini_train/include/core/runtime/runtime_common.h"
#include "infini_train/include/device.h"

#include "common/common_mlu.h"
#include "runtime/mlu_runtime_common.h"

namespace infini_train::core::mlu {
namespace {
constexpr int kMaxGpus = 8;
constexpr size_t kBytesPerMB = 1024ULL * 1024ULL;

static std::array<std::unique_ptr<MluStream>, kMaxGpus> mlu_queues;
static thread_local std::array<Stream *, kMaxGpus> current_queues = {};
#ifdef USE_CNNL
// Each thread owns one handle per device. Idle handles remain in the pool for reuse.
struct BlasHandlePool {
    std::mutex mutex;
    std::array<std::vector<std::shared_ptr<MluBlasHandle>>, kMaxGpus> handles;
};
static BlasHandlePool mlu_blas_handle_pool;
static thread_local std::array<std::shared_ptr<MluBlasHandle>, kMaxGpus> mlu_blas_handles;
#endif // USE_CNNL

static std::array<std::once_flag, kMaxGpus> device_stream_flags;

inline void CheckMluDevice(Device device) {
    CHECK(device.type() == Device::DeviceType::kPrivateUse1)
        << "MluGuardImpl expects PrivateUse1 (MLU) device, but got type=" << static_cast<int>(device.type())
        << " index=" << static_cast<int>(device.index());
    const int idx = device.index();
    CHECK(idx >= 0 && idx < kMaxGpus) << "MLU device index " << idx << " out of cache range [0, " << kMaxGpus << ").";
}

inline cnrtNotifier_t GetMluNotifier(Event *event) {
    auto *mlu_event = dynamic_cast<MluEvent *>(event);
    CHECK_NOTNULL(mlu_event);
    return mlu_event->mlu_notifier();
}

inline cnrtQueue_t GetMluQueue(Stream *stream) {
    auto *mlu_stream = dynamic_cast<MluStream *>(stream);
    CHECK_NOTNULL(mlu_stream);
    return mlu_stream->mlu_queue();
}

inline cnrtMemTransDir_t ToCnrtMemcpyKind(MemcpyKind kind) {
    switch (kind) {
    case MemcpyKind::kH2D:
        return cnrtMemcpyHostToDev;
    case MemcpyKind::kD2H:
        return cnrtMemcpyDevToHost;
    case MemcpyKind::kD2D:
        return cnrtMemcpyDevToDev;
    default:
        LOG(FATAL) << "Invalid MemcpyKind=" << MemcpyKindToString(kind);
    }
    return cnrtMemcpyDevToDev; // unreachable
}
} // namespace

void MluGuardImpl::InitSingleStream(Device device) {
    CheckMluDevice(device);

    int current_device = -1;
    MLU_CNRT_CHECK(cnrtGetDevice(&current_device));
    MLU_CNRT_CHECK(cnrtSetDevice(device.index()));

    mlu_queues[device.index()] = std::make_unique<MluStream>();
    current_queues[device.index()] = mlu_queues[device.index()].get();

    MLU_CNRT_CHECK(cnrtSetDevice(current_device));
}

void MluGuardImpl::InitSingleHandle(Device device) {
#ifdef USE_CNNL
    CheckMluDevice(device);

    auto &handle = mlu_blas_handles.at(device.index());
    if (handle) {
        return;
    }
    std::lock_guard<std::mutex> lock(mlu_blas_handle_pool.mutex);
    auto &pooled_handles = mlu_blas_handle_pool.handles.at(device.index());
    for (const auto &pooled_handle : pooled_handles) {
        // A count of one means only the pool owns the handle, so it is available.
        if (pooled_handle.use_count() == 1) {
            handle = pooled_handle;
            return;
        }
    }

    int current_device = -1;
    MLU_CNRT_CHECK(cnrtGetDevice(&current_device));
    MLU_CNRT_CHECK(cnrtSetDevice(device.index()));

    std::call_once(device_stream_flags.at(device.index()), InitSingleStream, device);

    handle = std::make_shared<MluBlasHandle>(mlu_queues[device.index()].get());
    pooled_handles.push_back(handle);

    MLU_CNRT_CHECK(cnrtSetDevice(current_device));
#endif // USE_CNNL
}

MluGuardImpl::MluGuardImpl() {}

// device
Device MluGuardImpl::GetDevice() const {
    int current_device = -1;
    MLU_CNRT_CHECK(cnrtGetDevice(&current_device));
    return Device(Device::DeviceType::kPrivateUse1, current_device);
}

void MluGuardImpl::SetDevice(Device device) const {
    CheckMluDevice(device);
    MLU_CNRT_CHECK(cnrtSetDevice(device.index()));
}

int MluGuardImpl::DeviceCount() const {
    unsigned int device_count = 0;
    MLU_CNRT_CHECK(cnrtGetDeviceCount(&device_count));
    return static_cast<int>(device_count);
}

Device::DeviceType MluGuardImpl::Type() const { return Device::DeviceType::kPrivateUse1; }

// stream
Stream *MluGuardImpl::GetStream(Device device) const {
    CheckMluDevice(device);
    std::call_once(device_stream_flags.at(device.index()), InitSingleStream, device);
    if (current_queues.at(device.index()) == nullptr) {
        current_queues.at(device.index()) = mlu_queues.at(device.index()).get();
    }
    return current_queues.at(device.index());
}

Stream *MluGuardImpl::CreateStream(Device device) const {
    CheckMluDevice(device);
    int current_device = -1;
    MLU_CNRT_CHECK(cnrtGetDevice(&current_device));
    MLU_CNRT_CHECK(cnrtSetDevice(device.index()));

    Stream *stream = new MluStream();

    MLU_CNRT_CHECK(cnrtSetDevice(current_device));
    return stream;
}

Stream *MluGuardImpl::CreateStreamWithPriority(Device device, int priority) const {
    CheckMluDevice(device);
    int current_device = -1;
    MLU_CNRT_CHECK(cnrtGetDevice(&current_device));
    MLU_CNRT_CHECK(cnrtSetDevice(device.index()));

    Stream *stream = new MluStream(priority);

    MLU_CNRT_CHECK(cnrtSetDevice(current_device));
    return stream;
}

void MluGuardImpl::DestroyStream(Stream *stream) const {
    if (stream == nullptr) {
        return;
    }
    auto *mlu_stream = dynamic_cast<MluStream *>(stream);
    CHECK_NOTNULL(mlu_stream);
    delete mlu_stream;
}

void MluGuardImpl::GetStreamPriorityRange(int *low, int *high) const {
    // This backend reports a single supported stream priority.
    *low = 0;
    *high = 0;
}

// event
void MluGuardImpl::EventCreate(Event **event) const { *event = new MluEvent(); }

void MluGuardImpl::EventCreateWithFlags(Event **event, EventFlag flags) const { *event = new MluEvent(flags); }

void MluGuardImpl::EventDestroy(Event *event) const {
    if (event == nullptr) {
        return;
    }
    delete event;
}

void MluGuardImpl::EventRecord(Event *event, Stream *stream) const {
    auto notifier = GetMluNotifier(event);
    auto queue = GetMluQueue(stream);
    MLU_CNRT_CHECK(cnrtPlaceNotifier(notifier, queue));
}

void MluGuardImpl::StreamWaitEvent(Stream *stream, Event *event, uint32_t flags) const {
    auto notifier = GetMluNotifier(event);
    auto queue = GetMluQueue(stream);
    MLU_CNRT_CHECK(cnrtQueueWaitNotifier(notifier, queue, flags));
}

RuntimeStatus MluGuardImpl::EventSynchronize(Event *event) const {
    auto notifier = GetMluNotifier(event);
    cnrtRet_t status = cnrtWaitNotifier(notifier);
    if (status == cnrtSuccess) {
        return RuntimeStatus::kSuccess;
    }
    LOG(ERROR) << "MluGuardImpl::EventSynchronize failed: " << cnrtGetErrorStr(status);
    return RuntimeStatus::kError;
}

RuntimeStatus MluGuardImpl::EventQuery(Event *event) const {
    auto notifier = GetMluNotifier(event);
    cnrtRet_t status = cnrtQueryNotifier(notifier);
    if (status == cnrtSuccess) {
        return RuntimeStatus::kSuccess;
    }
    if (status == cnrtErrorNotReady) {
        return RuntimeStatus::kNotReady;
    }
    LOG(ERROR) << "MluGuardImpl::EventQuery failed: " << cnrtGetErrorStr(status);
    return RuntimeStatus::kError;
}

float MluGuardImpl::EventElapsedTime(Event *start_event, Event *stop_event) const {
    auto start_notifier = GetMluNotifier(start_event);
    auto stop_notifier = GetMluNotifier(stop_event);
    float elapsed_ms = 0.0f;
    MLU_CNRT_CHECK(cnrtNotifierElapsedTime(start_notifier, stop_notifier, &elapsed_ms));
    return elapsed_ms;
}

// sync
void MluGuardImpl::SynchronizeDevice(Device device) const {
    auto original_device = GetDevice();
    SetDevice(device);

    MLU_CNRT_CHECK(cnrtSyncDevice());

    SetDevice(original_device);
}

void MluGuardImpl::SynchronizeStream(Stream *stream) const {
    auto queue = GetMluQueue(stream);
    MLU_CNRT_CHECK(cnrtQueueSync(queue));
}

// blas
BlasHandle *MluGuardImpl::GetBlasHandle(Device device) const {
#ifdef USE_CNNL
    CheckMluDevice(device);
    if (!mlu_blas_handles.at(device.index())) {
        InitSingleHandle(device);
    }
    auto *handle = mlu_blas_handles.at(device.index()).get();
    // Rebind to the current queue before returning the handle.
    MLU_CNNL_CHECK(cnnlSetQueue(handle->cnnl_handle(), GetMluQueue(GetStream(device))));
    return handle;
#else
    LOG(FATAL) << "MluGuardImpl::GetBlasHandle requires CNNL (build with -DINFINITRAIN_MLU_WITH_CNNL=ON "
                  "after installing the CNNL package)";
    return nullptr; // unreachable
#endif
}

// memory
void MluGuardImpl::Malloc(void **dev_ptr, size_t size) { MLU_CNRT_CHECK(cnrtMalloc(dev_ptr, size)); }

void MluGuardImpl::MallocAsync(void **dev_ptr, size_t size, Stream *stream) {
    // Use synchronous allocation because this CNRT version has no asynchronous allocator.
    MLU_CNRT_CHECK(cnrtMalloc(dev_ptr, size));
}

void MluGuardImpl::Free(void *dev_ptr) { MLU_CNRT_CHECK(cnrtFree(dev_ptr)); }

void MluGuardImpl::FreeAsync(void *dev_ptr, Stream *stream) {
    // Drain queued work before freeing memory it may still access.
    auto queue = GetMluQueue(stream);
    MLU_CNRT_CHECK(cnrtQueueSync(queue));
    MLU_CNRT_CHECK(cnrtFree(dev_ptr));
}

void MluGuardImpl::Memcpy(void *dst, const void *src, size_t count, MemcpyKind kind) {
    // Copy on the current stream and wait for completion so preceding kernels finish before host access.
    auto queue = GetMluQueue(GetStream(GetDevice()));
    if (queue != nullptr) {
        MLU_CNRT_CHECK(cnrtMemcpyAsync(dst, const_cast<void *>(src), count, queue, ToCnrtMemcpyKind(kind)));
        MLU_CNRT_CHECK(cnrtQueueSync(queue));
        return;
    }
    MLU_CNRT_CHECK(cnrtMemcpy(dst, const_cast<void *>(src), count, ToCnrtMemcpyKind(kind)));
}

void MluGuardImpl::MemcpyAsync(void *dst, const void *src, size_t count, MemcpyKind kind, Stream *stream) {
    auto queue = GetMluQueue(stream);
    MLU_CNRT_CHECK(cnrtMemcpyAsync(dst, const_cast<void *>(src), count, queue, ToCnrtMemcpyKind(kind)));
}

void MluGuardImpl::ResetMemPoolHighWatermarks(Device /*device*/) const {
    // CNRT exposes no resettable memory-pool high-water mark.
}

std::pair<size_t, size_t> MluGuardImpl::GetMemPoolPeakMB(Device device) const {
    // Report device-wide used memory as total minus free; there is no separate pool.
    CheckMluDevice(device);
    const auto original_device = GetDevice();
    SetDevice(device);
    size_t free_bytes = 0, total_bytes = 0;
    MLU_CNRT_CHECK(cnrtMemGetInfo(&free_bytes, &total_bytes));
    SetDevice(original_device);
    const size_t used_mb = (total_bytes - free_bytes) / kBytesPerMB;
    return {used_mb, used_mb};
}

INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL(Device::DeviceType::kPrivateUse1, MluGuardImpl)

} // namespace infini_train::core::mlu
