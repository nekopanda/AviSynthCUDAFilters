
#define _CRT_SECURE_NO_WARNINGS

#include <Windows.h>

#define AVS_LINKAGE_DLLIMPORT
#include "avisynth.h"
#pragma comment(lib, "avisynth.lib")

#include "gtest/gtest.h"

#include <fstream>
#include <string>
#include <iostream>

std::string GetDirectoryName(const std::string& filename)
{
	std::string directory;
	const size_t last_slash_idx = filename.rfind('\\');
	if (std::string::npos != last_slash_idx)
	{
		directory = filename.substr(0, last_slash_idx);
	}
	return directory;
}

// テスト対象となるクラス Foo のためのフィクスチャ
class TestBase : public ::testing::Test {
protected:
	TestBase() { }

	virtual ~TestBase() {
		// テスト毎に実行される，例外を投げない clean-up をここに書きます．
	}

	// コンストラクタとデストラクタでは不十分な場合．
	// 以下のメソッドを定義することができます：

	virtual void SetUp() {
		// このコードは，コンストラクタの直後（各テストの直前）
		// に呼び出されます．
		char buf[MAX_PATH];
		GetModuleFileName(nullptr, buf, MAX_PATH);
		modulePath = GetDirectoryName(buf);
		workDirPath = GetDirectoryName(GetDirectoryName(modulePath)) + "\\TestScripts";
	}

	virtual void TearDown() {
		// このコードは，各テストの直後（デストラクタの直前）
		// に呼び出されます．
	}

	std::string modulePath;
	std::string workDirPath;
};

TEST_F(TestBase, AnalyzeCUDATest)
{
	try {
		IScriptEnvironment2* env = CreateScriptEnvironment2();

		AVSValue result;
		std::string kdeintPath = modulePath + "\\KDeint.dll";
		env->LoadPlugin(kdeintPath.c_str(), true, &result);

		std::string script = "LWLibavVideoSource(\"test.ts\")\n"
			"s = KMSuper()\n"
			"kap = s.KMPartialSuper().KMAnalyse(isb = true, delta = 1, blksize = 16, overlap = 8, lambda = 400, global = true, meander = false)\n"
			"karef = s.KMAnalyse(isb = true, delta = 1, blksize = 16, overlap = 8, lambda = 400, global = true, meander = false, partial = kap)\n"
			"kacuda = s.OnCPU(0).KMAnalyse(isb = true, delta = 1, blksize = 16, overlap = 8, lambda = 400, global = true, meander = false, partial = kap.OnCPU(0)).OnCUDA(0)\n"
			"KMAnalyzeCheck2(karef, kacuda, last)";

		std::string scriptpath = workDirPath + "\\script.avs";

		std::ofstream out(scriptpath);
		out << script;
		out.close();

		{
			PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
			clip->GetFrame(100, env);
		}

		env->DeleteScriptEnvironment();
	}
	catch (const AvisynthError& err) {
		printf("%s\n", err.msg);
		GTEST_FAIL();
	}
}

int main(int argc, char **argv)
{
	::testing::GTEST_FLAG(filter) = "*AnalyzeCUDATest";
	::testing::InitGoogleTest(&argc, argv);
	int result = RUN_ALL_TESTS();

	getchar();

	return result;
}

