#include "infini_train_mlu/backend.h"

#include <mutex>
#include <optional>

#include "infini_train/include/core/privateuse1_backend.h"
#include "infini_train/include/datatype.h"

namespace infini_train::backends::mlu {

void RegisterBackend() {
    static std::once_flag once;
    std::call_once(once, [] {
        // The device guard and required kernels must already be statically registered.
        // Link the provider archives with the framework whole-archive helper.
        core::PrivateUse1BackendRegistration registration;
        registration.name = "mlu";
        registration.default_autocast_dtype = DataType::kBFLOAT16;
        core::RegisterPrivateUse1Backend(registration);
    });
}

} // namespace infini_train::backends::mlu
