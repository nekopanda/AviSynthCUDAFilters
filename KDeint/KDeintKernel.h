#pragma once

#include <windows.h>
#include "avisynth.h"

#include "CommonFunctions.h"

class KDeintKernel
{
  IScriptEnvironment* env;
  cudaStream_t stream;
public:

  void SetEnv(cudaStream_t stream, IScriptEnvironment* env)
  {
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

  template <typename pixel_t>
  void Copy(pixel_t* dst, int dst_pitch, const pixel_t* src, int src_pitch, int width, int height);

  template<typename pixel_t>
  void PadFrame(pixel_t *refFrame, int refPitch, int hPad, int vPad, int width, int height);

  template<typename pixel_t>
  void VerticalWiener(
    pixel_t *pDst, const pixel_t *pSrc, int nDstPitch,
    int nSrcPitch, int nWidth, int nHeight, int bits_per_pixel);

  template<typename pixel_t>
  void HorizontalWiener(
    pixel_t *pDst, const pixel_t *pSrc, int nDstPitch,
    int nSrcPitch, int nWidth, int nHeight, int bits_per_pixel);
  
  template<typename pixel_t>
  void RB2BilinearFiltered(
    pixel_t *pDst, const pixel_t *pSrc, int nDstPitch, int nSrcPitch, int nWidth, int nHeight);
};
