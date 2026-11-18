#include "runtime/mlu_runtime_common.h"

#include "glog/logging.h"

#include "common/common_mlu.h"

namespace infini_train::core::mlu {

MluEvent::MluEvent(EventFlag /*flags*/) {
    // CNRT notifier creation has no flags; wait/query implement the event semantics.
    MLU_CNRT_CHECK(cnrtNotifierCreate(&notifier_));
}

MluEvent::~MluEvent() {
    if (notifier_ != nullptr) {
        MLU_CNRT_CHECK(cnrtNotifierDestroy(notifier_));
    }
}

cnrtNotifier_t MluEvent::mlu_notifier() const { return notifier_; }

MluStream::MluStream() { MLU_CNRT_CHECK(cnrtQueueCreate(&queue_)); }

MluStream::MluStream(int priority) { MLU_CNRT_CHECK(cnrtQueueCreateWithPriority(&queue_, 0, priority)); }

MluStream::~MluStream() {
    // Do nothing.
}

cnrtQueue_t MluStream::mlu_queue() const { return queue_; }

#ifdef USE_CNNL
MluBlasHandle::MluBlasHandle(Stream *stream) {
    MLU_CNNL_CHECK(cnnlCreate(&cnnl_handle_));
    // Bind the handle to the current queue to preserve stream ordering.
    MLU_CNNL_CHECK(cnnlSetQueue(cnnl_handle_, dynamic_cast<MluStream *>(stream)->mlu_queue()));
}

MluBlasHandle::~MluBlasHandle() {
    // Do nothing.
}

cnnlHandle_t MluBlasHandle::cnnl_handle() const { return cnnl_handle_; }
#endif // USE_CNNL

} // namespace infini_train::core::mlu
