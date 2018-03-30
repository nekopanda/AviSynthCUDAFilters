#pragma once

#include "avisynth.h"

static bool IsCUDA(IScriptEnvironment* env_) {
   IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
   return env->GetProperty(AEP_DEVICE_TYPE) == DEV_TYPE_CUDA;
}
