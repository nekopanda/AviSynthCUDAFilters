#pragma once

#include <avisynth.h>

void Copy_(BYTE* dstp, int dst_pitch, const BYTE* srcp, int src_pitch, int row_size, int height, PNeoEnv env);

template <typename T>
static void Copy(T* dstp, int dst_pitch, const T* srcp, int src_pitch, int row_size, int height, PNeoEnv env) {
  Copy_(dstp, dst_pitch * sizeof(T), srcp, src_pitch * sizeof(T), src_pitch * sizeof(T), height, env);
}
