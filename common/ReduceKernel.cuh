#pragma once

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

template <typename T>
struct AddReducer {
	__device__ void operator()(T& v, T o) { v += o; }
};

template <typename T>
struct MaxReducer {
  __device__ void operator()(T& v, T o) { v = max(v, o); }
};

template <typename T>
struct MaxIndexReducer {
	__device__ void operator()(T& cnt, int& idx, T ocnt, int oidx) {
    if (ocnt > cnt) {
      cnt = ocnt;
      idx = oidx;
    }
  }
};

// MAX‚Í<=32‚©‚Â2‚×‚«‚Ì‚İ‘Î‰
template <typename T, int MAX, typename REDUCER>
__device__ void dev_reduce_warp(int tid, T& value)
{
  REDUCER red;
  // warp shuffle‚Åreduce
#if CUDART_VERSION >= 9000
  if (MAX >= 32) red(value, __shfl_down_sync(0xffffffff, value, 16));
  if (MAX >= 16) red(value, __shfl_down_sync(0xffffffff, value, 8));
  if (MAX >= 8) red(value, __shfl_down_sync(0xffffffff, value, 4));
  if (MAX >= 4) red(value, __shfl_down_sync(0xffffffff, value, 2));
  if (MAX >= 2) red(value, __shfl_down_sync(0xffffffff, value, 1));
#else
  if (MAX >= 32) red(value, __shfl_down(value, 16));
  if (MAX >= 16) red(value, __shfl_down(value, 8));
  if (MAX >= 8) red(value, __shfl_down(value, 4));
  if (MAX >= 4) red(value, __shfl_down(value, 2));
  if (MAX >= 2) red(value, __shfl_down(value, 1));
#endif
}

// MAX‚Í2‚×‚«‚Ì‚İ‘Î‰
// buf‚Íshared memory„§
template <typename T, int MAX, typename REDUCER>
__device__ void dev_reduce(int tid, T& value, T* buf)
{
  REDUCER red;
  if (MAX >= 64) {
    buf[tid] = value;
    __syncthreads();
    if (MAX >= 1024) {
      if (tid < 512) {
        red(buf[tid], buf[tid + 512]);
      }
      __syncthreads();
    }
    if (MAX >= 512) {
      if (tid < 256) {
        red(buf[tid], buf[tid + 256]);
      }
      __syncthreads();
    }
    if (MAX >= 256) {
      if (tid < 128) {
        red(buf[tid], buf[tid + 128]);
      }
      __syncthreads();
    }
    if (MAX >= 128) {
      if (tid < 64) {
        red(buf[tid], buf[tid + 64]);
      }
      __syncthreads();
    }
    if (MAX >= 64) {
      if (tid < 32) {
        red(buf[tid], buf[tid + 32]);
      }
      __syncthreads();
    }
    value = buf[tid];
  }
  if (tid < 32) {
    dev_reduce_warp<T, MAX, REDUCER>(tid, value);
  }
}

// MAX‚Í<=32‚©‚Â2‚×‚«‚Ì‚İ‘Î‰
template <typename K, typename V, int MAX, typename REDUCER>
__device__ void dev_reduce2_warp(int tid, K& key, V& value)
{
  REDUCER red;
  if (MAX >= 32) {
#if CUDART_VERSION >= 9000
    K okey = __shfl_down_sync(0xffffffff, key, 16);
    V ovalue = __shfl_down_sync(0xffffffff, value, 16);
#else
    K okey = __shfl_down(key, 16);
    V ovalue = __shfl_down(value, 16);
#endif
    red(key, value, okey, ovalue);
  }
  if (MAX >= 16) {
#if CUDART_VERSION >= 9000
    K okey = __shfl_down_sync(0xffffffff, key, 8);
    V ovalue = __shfl_down_sync(0xffffffff, value, 8);
#else
    K okey = __shfl_down(key, 8);
    V ovalue = __shfl_down(value, 8);
#endif
    red(key, value, okey, ovalue);
  }
  if (MAX >= 8) {
#if CUDART_VERSION >= 9000
    K okey = __shfl_down_sync(0xffffffff, key, 4);
    V ovalue = __shfl_down_sync(0xffffffff, value, 4);
#else
    K okey = __shfl_down(key, 4);
    V ovalue = __shfl_down(value, 4);
#endif
    red(key, value, okey, ovalue);
  }
  if (MAX >= 4) {
#if CUDART_VERSION >= 9000
    K okey = __shfl_down_sync(0xffffffff, key, 2);
    V ovalue = __shfl_down_sync(0xffffffff, value, 2);
#else
    K okey = __shfl_down(key, 2);
    V ovalue = __shfl_down(value, 2);
#endif
    red(key, value, okey, ovalue);
  }
  if (MAX >= 2) {
#if CUDART_VERSION >= 9000
    K okey = __shfl_down_sync(0xffffffff, key, 1);
    V ovalue = __shfl_down_sync(0xffffffff, value, 1);
#else
    K okey = __shfl_down(key, 1);
    V ovalue = __shfl_down(value, 1);
#endif
    red(key, value, okey, ovalue);
  }
}

// MAX‚Í2‚×‚«‚Ì‚İ‘Î‰
template <typename K, typename V, int MAX, typename REDUCER>
__device__ void dev_reduce2(int tid, K& key, V& value, K* kbuf, V* vbuf)
{
  REDUCER red;
  if (MAX >= 64) {
    kbuf[tid] = key;
    vbuf[tid] = value;
    __syncthreads();
    if (MAX >= 1024) {
      if (tid < 512) {
        red(kbuf[tid], vbuf[tid], kbuf[tid + 512], vbuf[tid + 512]);
      }
      __syncthreads();
    }
    if (MAX >= 512) {
      if (tid < 256) {
        red(kbuf[tid], vbuf[tid], kbuf[tid + 256], vbuf[tid + 256]);
      }
      __syncthreads();
    }
    if (MAX >= 256) {
      if (tid < 128) {
        red(kbuf[tid], vbuf[tid], kbuf[tid + 128], vbuf[tid + 128]);
      }
      __syncthreads();
    }
    if (MAX >= 128) {
      if (tid < 64) {
        red(kbuf[tid], vbuf[tid], kbuf[tid + 64], vbuf[tid + 64]);
      }
      __syncthreads();
    }
    if (MAX >= 64) {
      if (tid < 32) {
        red(kbuf[tid], vbuf[tid], kbuf[tid + 32], vbuf[tid + 32]);
      }
      __syncthreads();
    }
    key = kbuf[tid];
    value = vbuf[tid];
  }
  if (tid < 32) {
    dev_reduce2_warp<K, V, MAX, REDUCER>(tid, key, value);
  }
}

// MAX‚Í<=32‚©‚Â2‚×‚«‚Ì‚İ‘Î‰
template <typename T, int MAX, typename REDUCER>
__device__ void dev_scan_warp(int tid, T& value)
{
  REDUCER red;
  // warp shuffle‚Åscan
  if (MAX >= 2) {
#if CUDART_VERSION >= 9000
    T tmp = __shfl_up_sync(0xffffffff, value, 1);
#else
    T tmp = __shfl_up(value, 1);
#endif
    if (tid >= 1) red(value, tmp);
  }
  if (MAX >= 4) {
#if CUDART_VERSION >= 9000
    T tmp = __shfl_up_sync(0xffffffff, value, 2);
#else
    T tmp = __shfl_up(value, 2);
#endif
    if (tid >= 2) red(value, tmp);
  }
  if (MAX >= 8) {
#if CUDART_VERSION >= 9000
    T tmp = __shfl_up_sync(0xffffffff, value, 4);
#else
    T tmp = __shfl_up(value, 4);
#endif
    if (tid >= 4) red(value, tmp);
  }
  if (MAX >= 16) {
#if CUDART_VERSION >= 9000
    T tmp = __shfl_up_sync(0xffffffff, value, 8);
#else
    T tmp = __shfl_up(value, 8);
#endif
    if (tid >= 8) red(value, tmp);
  }
  if (MAX >= 32) {
#if CUDART_VERSION >= 9000
    T tmp = __shfl_up_sync(0xffffffff, value, 16);
#else
    T tmp = __shfl_up(value, 16);
#endif
    if (tid >= 16) red(value, tmp);
  }
}

// MAX‚Í2‚×‚«‚Ì‚İ‘Î‰
// buf‚Íshared memory„§ ’·‚³: MAX/32
template <typename T, int MAX, typename REDUCER>
__device__ void dev_scan(int tid, T& value, T* buf)
{
  REDUCER red;
  int wid = tid & 31;
  // ‚Ü‚¸warp“à‚Åscan
  dev_scan_warp<T, MAX, REDUCER>(wid, value);
  if (MAX >= 64) {
    // warp‚²‚Æ‚ÌŒ‹‰Ê‚ğsharedƒƒ‚ƒŠ‚ğ‰î‚µ‚ÄW–ñ
    if (wid == 31) buf[tid >> 5] = value;
    __syncthreads();
    if (tid < MAX / 32) {
      // warp‚²‚Æ‚ÌŒ‹‰Ê‚ğwarp“à‚Å‚³‚ç‚Éscan
      T v2 = buf[tid];
      dev_scan_warp<T, MAX / 32, REDUCER>(wid, v2);
      // sharedƒƒ‚ƒŠ‚ğ‰î‚µ‚Ä•ª”z
      buf[tid] = v2;
    }
    __syncthreads();
    // warp‚²‚Æ‚ÌscanŒ‹‰Ê‚ğ‘«‚·
    if(tid >= 32) red(value, buf[(tid >> 5) - 1]);
    __syncthreads();
  }
}

