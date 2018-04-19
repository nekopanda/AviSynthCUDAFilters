#define _CRT_SECURE_NO_WARNINGS

#define AVS_LINKAGE_DLLIMPORT
#include "avisynth.h"
#pragma comment(lib, "avisynth.lib")

#define NOMINMAX
#include <Windows.h>

#include "gtest/gtest.h"

int main(int argc, char **argv)
{
  ::testing::GTEST_FLAG(filter) = "*UCF2Perf*";
  ::testing::InitGoogleTest(&argc, argv);

  //_crtBreakAlloc = 7978;
  //_CrtMemState s1;
  //_CrtMemCheckpoint(&s1);

  int result = RUN_ALL_TESTS();

  //_CrtMemDumpAllObjectsSince(&s1);

  //getchar();

  return result;
}

