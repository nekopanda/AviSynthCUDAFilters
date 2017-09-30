#pragma once

#include <cassert>

#ifdef ENABLE_CUDA
#include <cuda_runtime_api.h>
#else
#define __host__
#define __device__
#endif

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#define PI 3.1415926535897932384626433832795

inline static int nblocks(int n, int block)
{
  return (n + block - 1) / block;
}

/* returns the biggest integer x such as 2^x <= i */
inline static int nlog2(int i)
{
#if 0
  int result = 0;
  while (i > 1) { i /= 2; result++; }
  return result;
#else
  assert(i > 0);
  unsigned long result;
  _BitScanReverse(&result, i);
  return result;
#endif
}

// CUDA‚Ìê‡‚ÍŠù‚É’è‹`‚³‚ê‚Ä‚¢‚éŠÖ”‚ªintrinsic‚ğŒÄ‚ñ‚Å‚­‚ê‚é‚Ì‚Å•K—v‚È‚¢
#ifndef __CUDA_ARCH__
template<typename T>
__host__ __device__ T min(T v1, T v2)
{
  return v1 < v2 ? v1 : v2;
}

template<typename T>
__host__ __device__ T max(T v1, T v2)
{
  return v1 > v2 ? v1 : v2;
}
#endif

template<typename T>
__host__ __device__ T clamp(T n, T minv, T maxv)
{
  return max(minv, min(n, maxv));
}

#ifdef ENABLE_CUDA
#define CUDA_CHECK(call) \
	do { \
		cudaError_t err__ = call; \
		if (err__ != cudaSuccess) { \
			OnCudaError(err__); \
			env->ThrowError("[CUDA Error] %d: %s", err__, cudaGetErrorString(err__)); \
				} \
		} while (0)

void OnCudaError(cudaError_t err);
#endif
