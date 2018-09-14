#pragma once

#include <avisynth.h>

void Copy_(BYTE* dstp, int dst_pitch, const BYTE* srcp, int src_pitch, int row_size, int height, PNeoEnv env);

template <typename T>
static void Copy(T* dstp, int dst_pitch, const T* srcp, int src_pitch, int row_size, int height, PNeoEnv env) {
  Copy_((BYTE*)dstp, dst_pitch * sizeof(T), (BYTE*)srcp, src_pitch * sizeof(T), row_size * sizeof(T), height, env);
}
