#pragma once
#include "avisynth.h"
#include <string>

template <typename pixel_t>
void DrawText(PVideoFrame &dst_, int bitsPerComponent, int x1, int y1, const std::string& s, PNeoEnv env);
