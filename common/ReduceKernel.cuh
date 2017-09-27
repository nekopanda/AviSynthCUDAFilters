#pragma once

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

template <typename T>
struct AddReducer {
	__device__ void operator()(T& v, T o) { v += o; }
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

// MAX��<=32����2�ׂ��̂ݑΉ�
template <typename T, int MAX, typename REDUCER>
__device__ void dev_reduce_warp(int tid, T& value)
{
  REDUCER red;
  // warp shuffle��reduce
  if (MAX >= 32) red(value, __shfl_down(value, 16));
  if (MAX >= 16) red(value, __shfl_down(value, 8));
  if (MAX >= 8) red(value, __shfl_down(value, 4));
  if (MAX >= 4) red(value, __shfl_down(value, 2));
  if (MAX >= 2) red(value, __shfl_down(value, 1));
}

// MAX��2�ׂ��̂ݑΉ�
// buf��shared memory����
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

// MAX��<=32����2�ׂ��̂ݑΉ�
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

// MAX��2�ׂ��̂ݑΉ�
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

// MAX��<=32����2�ׂ��̂ݑΉ�
template <typename T, int MAX, typename REDUCER>
__device__ void dev_scan_warp(int tid, T& value)
{
  REDUCER red;
  // warp shuffle��scan
  if (MAX >= 2) {
    T tmp = __shfl_up(value, 1);
    if (tid >= 1) red(value, tmp);
  }
  if (MAX >= 4) {
    T tmp = __shfl_up(value, 2);
    if (tid >= 2) red(value, tmp);
  }
  if (MAX >= 8) {
    T tmp = __shfl_up(value, 4);
    if (tid >= 4) red(value, tmp);
  }
  if (MAX >= 16) {
    T tmp = __shfl_up(value, 8);
    if (tid >= 8) red(value, tmp);
  }
  if (MAX >= 32) {
    T tmp = __shfl_up(value, 16);
    if (tid >= 16) red(value, tmp);
  }
}

// MAX��2�ׂ��̂ݑΉ�
// buf��shared memory���� ����: MAX/32
template <typename T, int MAX, typename REDUCER>
__device__ void dev_scan(int tid, T& value, T* buf)
{
  REDUCER red;
  int wid = tid & 31;
  // �܂�warp����scan
  dev_scan_warp<T, MAX, REDUCER>(wid, value);
  if (MAX >= 64) {
    // warp���Ƃ̌��ʂ�shared����������ďW��
    if (wid == 31) buf[tid >> 5] = value;
    __syncthreads();
    if (tid < MAX / 32) {
      // warp���Ƃ̌��ʂ�warp���ł����scan
      T v2 = buf[tid];
      dev_scan_warp<T, MAX / 32, REDUCER>(wid, v2);
      // shared����������ĕ��z
      buf[tid] = v2;
    }
    __syncthreads();
    // warp���Ƃ�scan���ʂ𑫂�
    if(tid >= 32) red(value, buf[(tid >> 5) - 1]);
    __syncthreads();
  }
}
