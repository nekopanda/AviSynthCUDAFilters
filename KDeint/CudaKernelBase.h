#pragma once

#include <windows.h>
#include "avisynth.h"

#include "CommonFunctions.h"

// CUDAƒJ[ƒlƒ‹ŽÀ‘•‚Ì‹¤’Êˆ—
class CudaKernelBase
{
protected:
  bool isEnabled;
  IScriptEnvironment* env;
  cudaStream_t stream;
public:

  void SetEnv(bool isEnabled, cudaStream_t stream, IScriptEnvironment* env)
  {
    this->isEnabled = isEnabled;
    this->env = env;
    this->stream = stream;
  }

  void DebugSync()
  {
#ifndef NDEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
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

