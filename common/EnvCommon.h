#pragma once

#include "avisynth.h"

static bool IsCUDA(PNeoEnv env) {
   return env->GetDeviceType() == DEV_TYPE_CUDA;
}
