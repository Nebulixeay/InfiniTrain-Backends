#pragma once

#ifdef USE_CNNL
#include <cnnl.h>
#endif
#include <cnrt.h>

#include "infini_train/include/core/runtime/runtime_common.h"

namespace infini_train::core::mlu {

class MluEvent final : public Event {
public:
    explicit MluEvent(EventFlag flags = EventFlag::kDefault);
    ~MluEvent() override;

    cnrtNotifier_t mlu_notifier() const;

private:
    cnrtNotifier_t notifier_ = nullptr;
};

class MluStream : public Stream {
public:
    MluStream();
    explicit MluStream(int priority);

    // Keep the global stream alive at process shutdown: CNRT may already be destroyed when static destructors run.
    ~MluStream() override;

    cnrtQueue_t mlu_queue() const;

private:
    cnrtQueue_t queue_ = nullptr;
};

#ifdef USE_CNNL
class MluBlasHandle : public BlasHandle {
public:
    explicit MluBlasHandle(Stream *stream);

    // Keep the pooled handle alive because CNNL teardown may precede static destruction.
    ~MluBlasHandle() override;

    cnnlHandle_t cnnl_handle() const;

private:
    cnnlHandle_t cnnl_handle_;
};
#endif // USE_CNNL

} // namespace infini_train::core::mlu
