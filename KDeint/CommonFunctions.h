#pragma once

#include <cuda_runtime_api.h>

#define PI 3.1415926535897932384626433832795

inline static int nblocks(int n, int block)
{
  return (n + block - 1) / block;
}

/* returns the biggest integer x such as 2^x <= i */
inline static int ilog2(int i)
{
  int result = 0;
  while (i > 1) { i /= 2; result++; }
  return result;
}

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

template<typename T>
__host__ __device__ T clamp(T n, T min, T max)
{
  n = n > max ? max : n;
  return n < min ? min : n;
}

#define CUDA_CHECK(call) \
	do { \
		cudaError_t err__ = call; \
		if (err__ != cudaSuccess) { \
			env->ThrowError("[CUDA Error] %d: %s", err__, cudaGetErrorString(err__)); \
				} \
		} while (0)


struct VECTOR
{
  int x;
  int y;
  int sad;
};

struct MVDataGroup
{
  int isValid;
  BYTE data[1]; // MVData[]
};

struct MVData
{
  int nCount;
  VECTOR data[1];
};
