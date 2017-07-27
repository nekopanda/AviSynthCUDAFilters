
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

template <typename T>
struct AddReducer {
  void operator()(T& v, T o) { v += o; }
};

template <typename T>
struct MaxIndexReducer {
  void operator()(T& cnt, int& idx, T ocnt, int oidx) {
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
  if (MAX >= 32) red(value, __shfl_down(value, 16));
  if (MAX >= 16) red(value, __shfl_down(value, 8));
  if (MAX >= 8) red(value, __shfl_down(value, 4));
  if (MAX >= 4) red(value, __shfl_down(value, 2));
  if (MAX >= 2) red(value, __shfl_down(value, 1));
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
    K okey = __shfl_down(key, 16);
    V ovalue = __shfl_down(value, 16);
    red(key, value, okey, ovalue);
  }
  if (MAX >= 16) {
    K okey = __shfl_down(key, 8);
    V ovalue = __shfl_down(value, 8);
    red(key, value, okey, ovalue);
  }
  if (MAX >= 8) {
    K okey = __shfl_down(key, 4);
    V ovalue = __shfl_down(value, 4);
    red(key, value, okey, ovalue);
  }
  if (MAX >= 4) {
    K okey = __shfl_down(key, 2);
    V ovalue = __shfl_down(value, 2);
    red(key, value, okey, ovalue);
  }
  if (MAX >= 2) {
    K okey = __shfl_down(key, 1);
    V ovalue = __shfl_down(value, 1);
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
    dev_reduce2_warp<T, MAX, REDUCER>(tid, key, value);
  }
}
