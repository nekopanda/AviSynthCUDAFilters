
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#include <Windows.h>

#define AVS_LINKAGE_DLLIMPORT
#include "avisynth.h"
#pragma comment(lib, "avisynth.lib")

#include "gtest/gtest.h"

int main(int argc, char **argv)
{
	::testing::GTEST_FLAG(filter) = "MaskToolsTest.Func";
	::testing::InitGoogleTest(&argc, argv);
	int result = RUN_ALL_TESTS();

	getchar();

	return result;
}

