#pragma once

#include "glog/logging.h"
#ifdef USE_CNCL
#include <cncl.h>
#endif
#ifdef USE_CNNL
#include <cnnl.h>
#endif
#include <cnrt.h>

namespace infini_train::common::mlu {

#define MLU_CNRT_CHECK(call)                                                                                           \
    do {                                                                                                               \
        cnrtRet_t status = (call);                                                                                     \
        if (status != cnrtSuccess) {                                                                                   \
            LOG(FATAL) << "CNRT Error: " << cnrtGetErrorStr(status) << " at " << __FILE__ << ":" << __LINE__;          \
        }                                                                                                              \
    } while (0)

#ifdef USE_CNNL
#define MLU_CNNL_CHECK(call)                                                                                           \
    do {                                                                                                               \
        cnnlStatus_t status = (call);                                                                                  \
        if (status != CNNL_STATUS_SUCCESS) {                                                                           \
            LOG(FATAL) << "CNNL Error: " << cnnlGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__;       \
        }                                                                                                              \
    } while (0)
#endif

#ifdef USE_CNCL
#define MLU_CNCL_CHECK(expr)                                                                                           \
    do {                                                                                                               \
        cnclResult_t _status = (expr);                                                                                 \
        if (_status != cnclSuccess) {                                                                                  \
            LOG(FATAL) << "CNCL error: " << cnclGetErrorStr(_status) << " at " << __FILE__ << ":" << __LINE__ << " ("  \
                       << #expr << ")";                                                                                \
        }                                                                                                              \
    } while (0)
#endif

} // namespace infini_train::common::mlu
