#pragma once

#include <windows.h>
#include "avisynth.h"

#include "CommonFunctions.h"

// CUDAƒJ[ƒlƒ‹ŽÀ‘•‚Ì‹¤’Êˆ—
class CudaKernelBase
{
protected:
  IScriptEnvironment2* env;
  cudaStream_t stream;
public:

  void SetEnv(cudaStream_t stream, IScriptEnvironment2* env)
  {
    this->env = env;
    this->stream = stream;
  }

  void DebugSync()
  {
#ifndef NDEBUG
//#if 1
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
  }

  void VerifyCUDAPointer(void* ptr)
  {
#ifndef NDEBUG
    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
    if (attr.memoryType != cudaMemoryTypeDevice) {
      env->ThrowError("[CUDA Error] Not valid devicce pointer");
    }
#endif
  }
};

