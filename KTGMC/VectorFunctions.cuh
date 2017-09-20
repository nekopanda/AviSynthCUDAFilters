#pragma once

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

// int拡張は明示的に書く //

// to_int(uchar4)
__device__ int4 to_int(uchar4 a) {
  int4 r = { a.x, a.y, a.z, a.w };
  return r;
}

// to_int(ushort4)
__device__ int4 to_int(ushort4 a) {
  int4 r = { a.x, a.y, a.z, a.w };
  return r;
}

// to_int(int4)
__device__ int4 to_int(int4 a) {
  return a;
}

__device__ int4 load_to_int(const unsigned char* p) {
  int4 r = { __ldg(&p[0]), __ldg(&p[1]), __ldg(&p[2]), __ldg(&p[3]) };
  return r;
}

__device__ int4 load_to_int(const unsigned short* p) {
  int4 r = { __ldg(&p[0]), __ldg(&p[1]), __ldg(&p[2]), __ldg(&p[3]) };
  return r;
}

// int4 + int
__device__ int4 operator+(int4 a, int b) {
  int4 r = { a.x + b, a.y + b, a.z + b, a.w + b };
  return r;
}

// int4 + int4
__device__ int4 operator+(int4 a, int4 b) {
  int4 r = { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
  return r;
}

// int4 * int
__device__ int4 operator*(int4 a, int b) {
  int4 r = { a.x * b, a.y * b, a.z * b, a.w * b };
  return r;
}

// int4 * short4
__device__ int4 operator*(int4 a, short4 b) {
  int4 r = { a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
  return r;
}

// int4 >> int
__device__ int4 operator >> (int4 a, int b) {
  int4 r = { a.x >> b, a.y >> b, a.z >> b, a.w >> b };
  return r;
}

// int4 += int4
__device__ void operator+=(int4& a, int4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}

// ushort4 += int4
__device__ void operator+=(ushort4& a, int4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}

// ushort4 += ushort4
__device__ void operator+=(ushort4& a, ushort4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}

// min(int4, int)
__device__ int4 min(int4 a, int b) {
  int4 r = { min(a.x, b), min(a.y, b), min(a.z, b), min(a.w, b) };
  return r;
}

// グローバル関数のオーバーロードで書けない処理 //

template <typename V> struct VHelper { };

template <> struct VHelper<unsigned char> {
  static __device__ unsigned char make(int a) {
    return (unsigned char)a;
  }
};

template <> struct VHelper<unsigned short> {
  static __device__ unsigned short make(int a) {
    return (unsigned short)a;
  }
};

template <> struct VHelper<uchar4> {
  static __device__ uchar4 make(int a) {
    typedef unsigned char uchar;
    uchar4 r = { (uchar)a, (uchar)a, (uchar)a, (uchar)a };
    return r;
  }
  static __device__ uchar4 cast_to(int4 a) {
    typedef unsigned char uchar;
    uchar4 r = { (uchar)a.x, (uchar)a.y, (uchar)a.z, (uchar)a.w };
    return r;
  }
};

template <> struct VHelper<ushort4> {
  static __device__ ushort4 make(int a) {
    typedef unsigned short ushort;
    ushort4 r = { (ushort)a, (ushort)a, (ushort)a, (ushort)a };
    return r;
  }
  static __device__ ushort4 cast_to(int4 a) {
    typedef unsigned short ushort;
    ushort4 r = { (ushort)a.x, (ushort)a.y, (ushort)a.z, (ushort)a.w };
    return r;
  }
};

template <> struct VHelper<int4> {
  static __device__ int4 make(int a) {
    int4 r = { a, a, a, a };
    return r;
  }
};


