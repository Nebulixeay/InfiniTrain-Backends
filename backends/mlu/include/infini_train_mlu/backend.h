#pragma once

// Register the MLU provider before parsing --device: Device::ParseType("mlu")
// requires its registered name. Upstream examples call this through the
// INFINITRAIN_EXAMPLE_EXTERNAL_BACKEND_HEADER / _REGISTRAR build definitions.

namespace infini_train::backends::mlu {

void RegisterBackend();

} // namespace infini_train::backends::mlu
