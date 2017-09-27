#pragma once

#include <stdint.h>

void GetWorkBytes(int width, int height, int& nnBytes, int& blockBytes);
void CopyPadCUDA(int pixelsize, uint8_t* ref, int refpitch, const uint8_t* src, int srcpitch, int width, int height, IScriptEnvironment2* env);
void BitBltCUDA(uint8_t* dst, int dstpitch, const uint8_t* src, int srcpitch, int width, int height, IScriptEnvironment2* env);
void EvalCUDA(int pixelsize, int bits_per_pixel,
  uint8_t* dst, int dstpitch, const uint8_t* ref, int refpitch, int width, int height,
  const int16_t* weights0, const int16_t* weights1, int weights1pitch,
  uint8_t* workNN, uint8_t* workBlock,
  int range_mode, int qual, int nns, int xdia, int ydia, IScriptEnvironment2* env);
