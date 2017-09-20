
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

  enum TEST_FRAMES {
    TF_MID, TF_BEGIN, TF_END
  };

  void GetFrames(PClip& clip, TEST_FRAMES tf, IScriptEnvironment2* env);

  void AnalyzeCUDATest(TEST_FRAMES tf, bool cuda, int blksize, bool chroma, int pel, int batch);
  void DegrainCUDATest(TEST_FRAMES tf, int N, int blksize, int pel);
  void CompensateCUDATest(TEST_FRAMES tf, int blksize, int pel);
};

void TestBase::GetFrames(PClip& clip, TEST_FRAMES tf, IScriptEnvironment2* env)
{
  int nframes = clip->GetVideoInfo().num_frames;
  switch (tf) {
  case TF_MID:
    for (int i = 0; i < 8; ++i) {
      clip->GetFrame(100 + i, env);
    }
    break;
  case TF_BEGIN:
    clip->GetFrame(0, env);
    clip->GetFrame(1, env);
    break;
  case TF_END:
    clip->GetFrame(nframes - 2, env);
    clip->GetFrame(nframes - 1, env);
    break;
  }
}

void TestBase::AnalyzeCUDATest(TEST_FRAMES tf, bool cuda, int blksize, bool chroma, int pel, int batch)
{
	try {
		IScriptEnvironment2* env = CreateScriptEnvironment2();

		AVSValue result;
		std::string kdeintPath = modulePath + "\\KDeint.dll";
		env->LoadPlugin(kdeintPath.c_str(), true, &result);

		std::string scriptpath = workDirPath + "\\script.avs";

		std::ofstream out(scriptpath);

		out << "LWLibavVideoSource(\"test.ts\")" << std::endl;
		out << "s = KMSuper(pel = " << pel << ")" << std::endl;
		out << "kap = s.KMPartialSuper().KMAnalyse(isb = true, delta = 1, chroma = " <<
			(chroma ? "true" : "false") << ", blksize = " << blksize <<
			", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false)" << std::endl;
		out << "karef = s.KMAnalyse(isb = true, delta = 1, chroma = " <<
			(chroma ? "true" : "false") << ", blksize = " << blksize <<
			", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false)" << std::endl;
		out << "kacuda = s." << (cuda ? "OnCPU(0)." : "") << "KMAnalyse(isb = true, delta = 1, chroma = " <<
			(chroma ? "true" : "false") << ", blksize = " << blksize <<
			", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false, batch = " << batch << ", partial = kap" <<
      (cuda ? ".OnCPU(0)" : "") << ")" << (cuda ? ".OnCUDA(0)" : "") << std::endl;
		out << "KMAnalyzeCheck2(karef, kacuda, last)" << std::endl;

		out.close();

		{
			PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      GetFrames(clip, tf, env);
		}

		env->DeleteScriptEnvironment();
	}
	catch (const AvisynthError& err) {
		printf("%s\n", err.msg);
		GTEST_FAIL();
	}
}

#pragma region Analyze

TEST_F(TestBase, AnalyzeCUDA_Blk32NoCPel1Batch1)
{
  AnalyzeCUDATest(TF_MID, true, 32, false, 1, 1);
}

TEST_F(TestBase, AnalyzeCUDA_Blk32NoCPel1Batch2)
{
  AnalyzeCUDATest(TF_MID, true, 32, false, 1, 2);
}

TEST_F(TestBase, AnalyzeCUDA_Blk32NoCPel1Batch3)
{
  AnalyzeCUDATest(TF_MID, true, 32, false, 1, 3);
}

TEST_F(TestBase, AnalyzeCUDA_Blk32NoCPel1Batch8)
{
  AnalyzeCUDATest(TF_MID, true, 32, false, 1, 8);
}

TEST_F(TestBase, AnalyzeCUDA_Blk16WithCPel2)
{
	AnalyzeCUDATest(TF_MID, true, 16, true, 2, 4);
}

TEST_F(TestBase, AnalyzeCUDA_Blk16WithCPel1)
{
	AnalyzeCUDATest(TF_MID, true, 16, true, 1, 4);
}

TEST_F(TestBase, AnalyzeCUDA_Blk16NoCPel2)
{
	AnalyzeCUDATest(TF_MID, true, 16, false, 2, 4);
}

TEST_F(TestBase, AnalyzeCUDA_Blk16NoCPel1)
{
	AnalyzeCUDATest(TF_MID, true, 16, false, 1, 4);
}

TEST_F(TestBase, AnalyzeCUDA_Blk32WithCPel2)
{
	AnalyzeCUDATest(TF_MID, true, 32, true, 2, 4);
}

TEST_F(TestBase, AnalyzeCUDA_Blk32WithCPel1)
{
	AnalyzeCUDATest(TF_MID, true, 32, true, 1, 4);
}

TEST_F(TestBase, AnalyzeCUDA_Blk32NoCPel2)
{
	AnalyzeCUDATest(TF_MID, true, 32, false, 2, 4);
}

TEST_F(TestBase, AnalyzeCUDA_Blk32NoCPel1)
{
	AnalyzeCUDATest(TF_MID, true, 32, false, 1, 4);
}

TEST_F(TestBase, AnalyzeCPU_Blk16WithCPel2)
{
  AnalyzeCUDATest(TF_MID, false, 16, true, 2, 4);
}

#pragma endregion

void TestBase::DegrainCUDATest(TEST_FRAMES tf, int N, int blksize, int pel)
{
  try {
    IScriptEnvironment2* env = CreateScriptEnvironment2();

    AVSValue result;
    std::string kdeintPath = modulePath + "\\KDeint.dll";
    env->LoadPlugin(kdeintPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;
    out << "s = src.KMSuper(pel = " << pel << ")" << std::endl;
    out << "scuda = s.OnCPU(0)" << std::endl;
    out << "pmvb = s.KMPartialSuper().KMAnalyse(isb = true, delta = 1, chroma = false, blksize = " << blksize <<
      ", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false)" << std::endl;
    out << "pmvf = s.KMPartialSuper().KMAnalyse(isb = false, delta = 1, chroma = false, blksize = " << blksize <<
      ", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false)" << std::endl;
    if (N >= 2) {
      out << "pmvb1 = s.KMPartialSuper().KMAnalyse(isb = true, delta = 2, chroma = false, blksize = " << blksize <<
        ", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false)" << std::endl;
      out << "pmvf1 = s.KMPartialSuper().KMAnalyse(isb = false, delta = 2, chroma = false, blksize = " << blksize <<
        ", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false)" << std::endl;
    }
    if (true) { // MV CUDA版
      out << "mvb = scuda.KMAnalyse(isb = true, delta = 1, chroma = false, blksize = " << blksize <<
        ", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false, partial = pmvb.OnCPU(0))" << std::endl;
      out << "mvf = scuda.KMAnalyse(isb = false, delta = 1, chroma = false, blksize = " << blksize <<
        ", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false, partial = pmvf.OnCPU(0))" << std::endl;
      if (N >= 2) {
        out << "mvb1 = scuda.KMAnalyse(isb = true, delta = 2, chroma = false, blksize = " << blksize <<
          ", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false, partial = pmvb1.OnCPU(0))" << std::endl;
        out << "mvf1 = scuda.KMAnalyse(isb = false, delta = 2, chroma = false, blksize = " << blksize <<
          ", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false, partial = pmvf1.OnCPU(0))" << std::endl;
      }
      if (N == 1) {
        out << "degref = src.KMDegrain1(s, mvb.OnCUDA(0), mvf.OnCUDA(0), thSAD = 640, thSCD1 = 180, thSCD2 = 98)" << std::endl;
        out << "degcuda = srcuda.KMDegrain1(scuda, mvb, mvf, thSAD = 640, thSCD1 = 180, thSCD2 = 98).OnCUDA(0)" << std::endl;
      }
      else if (N == 2) {
        out << "degref = src.KMDegrain2(s, mvb.OnCUDA(0), mvf.OnCUDA(0), mvb1.OnCUDA(0), mvf1.OnCUDA(0), thSAD = 640, thSCD1 = 180, thSCD2 = 98)" << std::endl;
        out << "degcuda = srcuda.KMDegrain2(scuda, mvb, mvf, mvb1, mvf1, thSAD = 640, thSCD1 = 180, thSCD2 = 98).OnCUDA(0)" << std::endl;
      }
    }
    else {
      out << "mvb = s.KMAnalyse(isb = true, delta = 1, chroma = false, blksize = " << blksize <<
        ", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false, partial = pmvb)" << std::endl;
      out << "mvf = s.KMAnalyse(isb = false, delta = 1, chroma = false, blksize = " << blksize <<
        ", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false, partial = pmvf)" << std::endl;
      if (N >= 2) {
        out << "mvb1 = s.KMAnalyse(isb = true, delta = 2, chroma = false, blksize = " << blksize <<
          ", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false, partial = pmvb1)" << std::endl;
        out << "mvf1 = s.KMAnalyse(isb = false, delta = 2, chroma = false, blksize = " << blksize <<
          ", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false, partial = pmvf1)" << std::endl;
      }
      if (N == 1) {
        out << "degref = src.KMDegrain1(s, mvb, mvf, thSAD = 640, thSCD1 = 180, thSCD2 = 98)" << std::endl;
        out << "degcuda = srcuda.KMDegrain1(scuda, mvb.OnCPU(0), mvf.OnCPU(0), thSAD = 640, thSCD1 = 180, thSCD2 = 98).OnCUDA(0)" << std::endl;
      }
      else if (N == 2) {
        out << "degref = src.KMDegrain2(s, mvb, mvf, mvb1, mvf1, thSAD = 640, thSCD1 = 180, thSCD2 = 98)" << std::endl;
        out << "degcuda = srcuda.KMDegrain2(scuda, mvb.OnCPU(0), mvf.OnCPU(0), mvb1.OnCPU(0), mvf1.OnCPU(0), thSAD = 640, thSCD1 = 180, thSCD2 = 98).OnCUDA(0)" << std::endl;
      }
    }
    out << "ImageCompare(degref, degcuda)" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      GetFrames(clip, tf, env);
    }

    env->DeleteScriptEnvironment();
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

#pragma region Degrain

TEST_F(TestBase, DegrainCUDA_1Blk16Pel2)
{
  DegrainCUDATest(TF_MID, 1, 16, 2);
}

TEST_F(TestBase, DegrainCUDA_1Blk16Pel1)
{
  DegrainCUDATest(TF_MID, 1, 16, 1);
}

TEST_F(TestBase, DegrainCUDA_1Blk32Pel2)
{
  DegrainCUDATest(TF_MID, 1, 32, 2);
}

TEST_F(TestBase, DegrainCUDA_1Blk32Pel1)
{
  DegrainCUDATest(TF_MID, 1, 32, 1);
}

TEST_F(TestBase, DegrainCUDA_2Blk16Pel2)
{
  DegrainCUDATest(TF_MID, 2, 16, 2);
}

TEST_F(TestBase, DegrainCUDA_2Blk16Pel1)
{
  DegrainCUDATest(TF_MID, 2, 16, 1);
}

TEST_F(TestBase, DegrainCUDA_2Blk32Pel2)
{
  DegrainCUDATest(TF_MID, 2, 32, 2);
}

TEST_F(TestBase, DegrainCUDA_2Blk32Pel1)
{
  DegrainCUDATest(TF_MID, 2, 32, 1);
}

#pragma endregion

void TestBase::CompensateCUDATest(TEST_FRAMES tf, int blksize, int pel)
{
  try {
    IScriptEnvironment2* env = CreateScriptEnvironment2();

    AVSValue result;
    std::string kdeintPath = modulePath + "\\KDeint.dll";
    env->LoadPlugin(kdeintPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;
    out << "s = src.KMSuper(pel = " << pel << ")" << std::endl;
    out << "scuda = s.OnCPU(0)" << std::endl;
    out << "pmvb = s.KMPartialSuper().KMAnalyse(isb = true, delta = 1, chroma = false, blksize = " << blksize <<
      ", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false)" << std::endl;
    if (true) { // MV CUDA版
      out << "mvb = scuda.KMAnalyse(isb = true, delta = 1, chroma = false, blksize = " << blksize <<
        ", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false, partial = pmvb.OnCPU(0))" << std::endl;
      out << "comref = src.KMCompensate(s, mvb.OnCUDA(0),thSCD1=1800,thSCD2=980)" << std::endl;
      out << "comcuda = srcuda.KMCompensate(scuda,mvb,thSCD1=1800,thSCD2=980).OnCUDA(0)" << std::endl;
    }
    else {
      out << "mvb = s.KMAnalyse(isb = true, delta = 1, chroma = false, blksize = " << blksize <<
        ", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false, partial = pmvb)" << std::endl;
      out << "comref = src.KMCompensate(s, mvb, thSCD1=180,thSCD2=98)" << std::endl;
      out << "comcuda = srcuda.KMCompensate(scuda, mvb.OnCPU(0), thSCD1=180,thSCD2=98).OnCUDA(0)" << std::endl;
    }
    out << "ImageCompare(comref, comcuda)" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      GetFrames(clip, tf, env);
    }

    env->DeleteScriptEnvironment();
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

#pragma region Compensate

TEST_F(TestBase, CompensateCUDA_Blk16Pel2)
{
  CompensateCUDATest(TF_MID, 16, 2);
}

TEST_F(TestBase, CompensateCUDA_Blk16Pel1)
{
  CompensateCUDATest(TF_MID, 16, 1);
}

TEST_F(TestBase, CompensateCUDA_Blk32Pel2)
{
  CompensateCUDATest(TF_MID, 32, 2);
}

TEST_F(TestBase, CompensateCUDA_Blk32Pel1)
{
  CompensateCUDATest(TF_MID, 32, 1);
}

#pragma endregion

int main(int argc, char **argv)
{
	::testing::GTEST_FLAG(filter) = "*Blk32Pel1*";
	::testing::InitGoogleTest(&argc, argv);
	int result = RUN_ALL_TESTS();

	//getchar();

	return result;
}

