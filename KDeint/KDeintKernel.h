#pragma once

#include <windows.h>
#include "avisynth.h"

#include "CommonFunctions.h"

enum {
	ANALYZE_MAX_BATCH = 8
};

template <typename pixel_t>
class IKDeintKernel
{
public:
  typedef typename std::conditional <sizeof(pixel_t) == 1, unsigned short, int>::type tmp_t;

  virtual bool IsEnabled() const = 0;

  virtual void MemCpy(void* dst, const void* src, int nbytes) = 0;
  virtual void Copy(pixel_t* dst, int dst_pitch, const pixel_t* src, int src_pitch, int width, int height) = 0;
  virtual void PadFrame(pixel_t *refFrame, int refPitch, int hPad, int vPad, int width, int height) = 0;
  virtual void VerticalWiener(
    pixel_t *pDst, const pixel_t *pSrc, int nDstPitch,
    int nSrcPitch, int nWidth, int nHeight, int bits_per_pixel) = 0;
  virtual void HorizontalWiener(
    pixel_t *pDst, const pixel_t *pSrc, int nDstPitch,
    int nSrcPitch, int nWidth, int nHeight, int bits_per_pixel) = 0;
  virtual void RB2BilinearFiltered(
    pixel_t *pDst, const pixel_t *pSrc, int nDstPitch, int nSrcPitch, int nWidth, int nHeight) = 0;

	// Analyze //
  virtual int GetSearchBlockSize() = 0;
  virtual int GetSearchBatchSize() = 0;
  virtual void EstimateGlobalMV(int batch, const short2* vectors, int vectorsPitch, int nBlkCount, short2* globalMV) = 0;
  virtual void InterpolatePrediction(
		int batch,
		const short2* src_vector, int srcVectorPitch, const int* src_sad, int srcSadPitch,
		short2* dst_vector, int dstVectorPitch, int* dst_sad, int dstSadPitch,
		int nSrcBlkX, int nSrcBlkY, int nDstBlkX, int nDstBlkY,
		int normFactor, int normov, int atotal, int aodd, int aeven) = 0;
  virtual void LoadMV(const VECTOR* in, short2* vectors, int* sads, int nBlkCount) = 0;
  virtual void StoreMV(VECTOR* out, const short2* vectors, const int* sads, int nBlkCount) = 0;
  virtual void WriteDefaultMV(VECTOR* dst, int nBlkCount, int verybigSAD) = 0;

  // 36 args
  virtual void Search(
		int batch, void* _searchbatch, 
		int searchType, int nBlkX, int nBlkY, int nBlkSize, int nLogScale,
		int nLambdaLevel, int lsad, int penaltyZero, int penaltyGlobal, int penaltyNew,
		int nPel, bool chroma, int nPad, int nBlkSizeOvr, int nExtendedWidth, int nExptendedHeight,
		const pixel_t** pSrcY, const pixel_t** pSrcU, const pixel_t** pSrcV,
		const pixel_t** pRefY, const pixel_t** pRefU, const pixel_t** pRefV,
		int nPitchY, int nPitchUV, int nImgPitchY, int nImgPitchUV,
		const short2* globalMV, short2* vectors, int vectorsPitch, int* sads, int sadPitch, void* blocks, int* prog, int* next) = 0;

	// Degrain //
  virtual void GetDegrainStructSize(int N, int& degrainBlock, int& degrainArg) = 0;

  //34 args
  virtual void Degrain(
    int N, int nWidth, int nHeight, int nBlkX, int nBlkY, int nPad, int nBlkSize, int nPel, int nBitsPerPixel,
    bool* enableYUV, bool* isUsableB, bool* isUsableF,
    int nTh1, int nTh2, int thSAD, int thSADC,
    const short* ovrwins, const short* overwinsUV,
    const VECTOR** mvB, const VECTOR** mvF,
    const pixel_t** pSrc, pixel_t** pDst, tmp_t** pTmp, const pixel_t** pRefB, const pixel_t** pRefF,
    int nPitchY, int nPitchUV,
    int nPitchSuperY, int nPitchSuperUV, int nImgPitchY, int nImgPitchUV,
    void* _degrainblock, void* _degrainarg, int* sceneChange) = 0;

  // Compensate //
  virtual int GetCompensateStructSize() = 0;

  //31 args
  virtual void Compensate(
    int nWidth, int nHeight, int nBlkX, int nBlkY, int nPad, int nBlkSize, int nPel, int nBitsPerPixel,
    int nTh1, int nTh2, int time256, int thSAD,
    const short* ovrwins, const short* overwinsUV, const VECTOR* mv,
    const pixel_t** pSrc, pixel_t** pDst, tmp_t** pTmp, const pixel_t** pRef,
    int nPitchY, int nPitchUV,
    int nPitchSuperY, int nPitchSuperUV, int nImgPitchY, int nImgPitchUV,
    void* _compensateblock, int* sceneChange) = 0;
};

class IKDeintCUDA
{
public:
  virtual void SetEnv(cudaStream_t stream, IScriptEnvironment2* env) = 0;
  virtual bool IsEnabled() const = 0;
  virtual IKDeintKernel<uint8_t>* get(uint8_t) = 0;
  virtual IKDeintKernel<uint16_t>* get(uint16_t) = 0;
};

IKDeintCUDA* CreateKDeintCUDA();
