#pragma once

#include <avisynth.h>

struct FuncDefinition {
  typedef IScriptEnvironment::ApplyFunc apply_func_t;

   const char* name;
   const char* params;
   apply_func_t func;
   void* user_data;

   FuncDefinition(void*)
     : FuncDefinition(nullptr, nullptr, nullptr, nullptr, nullptr) { }
   FuncDefinition(const char* name, const char* _not_used, const char* params, apply_func_t func)
     : FuncDefinition(name, _not_used, params, func, nullptr) { }
   FuncDefinition(const char* name, const char* _not_used, const char* params, apply_func_t func, void *user_data)
     : name(name), params(params), func(func), user_data(user_data) { }
};

#define BUILTIN_FUNC_PREFIX "dummy"

int GetDeviceTypes(const PClip& clip);

#ifndef NDEBUG
//#if 1
#define DEBUG_SYNC \
CUDA_CHECK(cudaGetLastError()); \
CUDA_CHECK(cudaDeviceSynchronize())
#else
#define DEBUG_SYNC
#endif

#define IS_CUDA (env->GetDeviceType() == DEV_TYPE_CUDA)

#include "Copy.h"
