#pragma once

enum {
  OVERLAP = 8,
  BLOCK_SIZE = OVERLAP * 2,
  VPAD = 4,
};

int GetDeviceType(const PClip& clip);
