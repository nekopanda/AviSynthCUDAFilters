#pragma once

#include <windows.h>
#include "avisynth.h"

#include "CommonFunctions.h"

class KDeintKernel
{
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

	bool IsEnabled() const {
		return isEnabled;
	}

  void DebugSync()
  {
#ifndef NDEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
  }

	void MemCpy(void* dst, const void* src, int nbytes);

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

	// Analyze //

	int GetSearchBlockSize();

	void EstimateGlobalMV(const short2* vectors, int nBlkCount, short2* globalMV);

	void InterpolatePrediction(
		const short2* src_vector, const int* src_sad,
		short2* dst_vector, int* dst_sad,
		int nSrcBlkX, int nSrcBlkY, int nDstBlkX, int nDstBlkY,
		int normFactor, int normov, int atotal, int aodd, int aeven);

	void LoadMV(const VECTOR* in, short2* vectors, int* sads, int nBlkCount);

	void WriteDefaultMV(VECTOR* dst, int nBlkCount, int verybigSAD);

	template <typename pixel_t>
	void Search(
		int nBlkX, int nBlkY, int nBlkSize, int nLogScale,
		int nLambdaLevel, int lsad, int penaltyZero, int penaltyGlobal, int penaltyNew,
		int nPel, int nPad, int nBlkSizeOvr, int nExtendedWidth, int nExptendedHeight,
		const pixel_t* pSrcY, const pixel_t* pSrcU, const pixel_t* pSrcV,
		const pixel_t* pRefY, const pixel_t* pRefU, const pixel_t* pRefV,
		int nPitchY, int nPitchUV, int nImgPitchY, int nImgPitchUV,
		const short2* globalMV, short2* vectors, int* sads, void* blocks);
};
