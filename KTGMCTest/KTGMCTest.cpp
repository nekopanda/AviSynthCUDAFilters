
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

  void AnalyzeTest(TEST_FRAMES tf, bool cuda, int blksize, bool chroma, int pel, int batch);
  void DegrainTest(TEST_FRAMES tf, int N, int blksize, int pel);
  void DegrainBinomialTest(TEST_FRAMES tf, int N, int blksize, int pel);
  void CompensateTest(TEST_FRAMES tf, int blksize, int pel);

  void BobTest(TEST_FRAMES tf, bool parity);
  void BinomialSoftenTest(TEST_FRAMES tf, int radius, bool chroma);
  void RemoveGrainTest(TEST_FRAMES tf, int mode, bool chroma);
  void GaussResizeTest(TEST_FRAMES tf, bool chroma);

  void InpandVerticalX2Test(TEST_FRAMES tf, bool chroma);
  void ExpandVerticalX2Test(TEST_FRAMES tf, bool chroma);
  void MakeDiffTest(TEST_FRAMES tf, bool chroma);
  void LogicTest(TEST_FRAMES tf, const char* mode, bool chroma);

  void BobShimmerFixesMergeTest(TEST_FRAMES tf, int rep, bool chroma);
  void VResharpenTest(TEST_FRAMES tf);
  void ResharpenTest(TEST_FRAMES tf);
  void LimitOverSharpenTest(TEST_FRAMES tf);
  void ToFullRangeTest(TEST_FRAMES tf, bool chroma);
  void TweakSearchClipTest(TEST_FRAMES tf, bool chroma);

  void MergeTest(TEST_FRAMES tf, bool chroma);

  void NNEDI3Test(TEST_FRAMES tf, bool chroma);
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

#pragma region Analyze

void TestBase::AnalyzeTest(TEST_FRAMES tf, bool cuda, int blksize, bool chroma, int pel, int batch)
{
	try {
		IScriptEnvironment2* env = CreateScriptEnvironment2();

		AVSValue result;
		std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
		env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

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

TEST_F(TestBase, Analyze_Blk32NoCPel1Batch1)
{
  AnalyzeTest(TF_MID, true, 32, false, 1, 1);
}

TEST_F(TestBase, Analyze_Blk32NoCPel1Batch2)
{
  AnalyzeTest(TF_MID, true, 32, false, 1, 2);
}

TEST_F(TestBase, Analyze_Blk32NoCPel1Batch3)
{
  AnalyzeTest(TF_MID, true, 32, false, 1, 3);
}

TEST_F(TestBase, Analyze_Blk32NoCPel1Batch8)
{
  AnalyzeTest(TF_MID, true, 32, false, 1, 8);
}

TEST_F(TestBase, Analyze_Blk16WithCPel2)
{
	AnalyzeTest(TF_MID, true, 16, true, 2, 4);
}

TEST_F(TestBase, Analyze_Blk16WithCPel1)
{
	AnalyzeTest(TF_MID, true, 16, true, 1, 4);
}

TEST_F(TestBase, Analyze_Blk16NoCPel2)
{
	AnalyzeTest(TF_MID, true, 16, false, 2, 4);
}

TEST_F(TestBase, Analyze_Blk16NoCPel1)
{
	AnalyzeTest(TF_MID, true, 16, false, 1, 4);
}

TEST_F(TestBase, Analyze_Blk32WithCPel2)
{
	AnalyzeTest(TF_MID, true, 32, true, 2, 4);
}

TEST_F(TestBase, Analyze_Blk32WithCPel1)
{
	AnalyzeTest(TF_MID, true, 32, true, 1, 4);
}

TEST_F(TestBase, Analyze_Blk32NoCPel2)
{
	AnalyzeTest(TF_MID, true, 32, false, 2, 4);
}

TEST_F(TestBase, Analyze_Blk32NoCPel1)
{
	AnalyzeTest(TF_MID, true, 32, false, 1, 4);
}

TEST_F(TestBase, AnalyzeCPU_Blk16WithCPel2)
{
  AnalyzeTest(TF_MID, false, 16, true, 2, 4);
}

#pragma endregion

#pragma region Degrain

void TestBase::DegrainTest(TEST_FRAMES tf, int N, int blksize, int pel)
{
  try {
    IScriptEnvironment2* env = CreateScriptEnvironment2();

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    // シーンチェンジ判定されるとテストできないのでしきい値は10倍にしてある

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
        out << "degref = src.KMDegrain1(s, mvb.OnCUDA(0), mvf.OnCUDA(0), thSAD = 6400, thSCD1 = 1800, thSCD2 = 980)" << std::endl;
        out << "degcuda = srcuda.KMDegrain1(scuda, mvb, mvf, thSAD = 6400, thSCD1 = 1800, thSCD2 = 980).OnCUDA(0)" << std::endl;
      }
      else if (N == 2) {
        out << "degref = src.KMDegrain2(s, mvb.OnCUDA(0), mvf.OnCUDA(0), mvb1.OnCUDA(0), mvf1.OnCUDA(0), thSAD = 6400, thSCD1 = 1800, thSCD2 = 980)" << std::endl;
        out << "degcuda = srcuda.KMDegrain2(scuda, mvb, mvf, mvb1, mvf1, thSAD = 6400, thSCD1 = 1800, thSCD2 = 980).OnCUDA(0)" << std::endl;
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
        out << "degref = src.KMDegrain1(s, mvb, mvf, thSAD = 6400, thSCD1 = 1800, thSCD2 = 980)" << std::endl;
        out << "degcuda = srcuda.KMDegrain1(scuda, mvb.OnCPU(0), mvf.OnCPU(0), thSAD = 6400, thSCD1 = 1800, thSCD2 = 980).OnCUDA(0)" << std::endl;
      }
      else if (N == 2) {
        out << "degref = src.KMDegrain2(s, mvb, mvf, mvb1, mvf1, thSAD = 6400, thSCD1 = 1800, thSCD2 = 980)" << std::endl;
        out << "degcuda = srcuda.KMDegrain2(scuda, mvb.OnCPU(0), mvf.OnCPU(0), mvb1.OnCPU(0), mvf1.OnCPU(0), thSAD = 6400, thSCD1 = 1800, thSCD2 = 980).OnCUDA(0)" << std::endl;
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

TEST_F(TestBase, Degrain_1Blk16Pel2)
{
  DegrainTest(TF_MID, 1, 16, 2);
}

TEST_F(TestBase, Degrain_1Blk16Pel1)
{
  DegrainTest(TF_MID, 1, 16, 1);
}

TEST_F(TestBase, Degrain_1Blk32Pel2)
{
  DegrainTest(TF_MID, 1, 32, 2);
}

TEST_F(TestBase, Degrain_1Blk32Pel1)
{
  DegrainTest(TF_MID, 1, 32, 1);
}

TEST_F(TestBase, Degrain_2Blk16Pel2)
{
  DegrainTest(TF_MID, 2, 16, 2);
}

TEST_F(TestBase, Degrain_2Blk16Pel1)
{
  DegrainTest(TF_MID, 2, 16, 1);
}

TEST_F(TestBase, Degrain_2Blk32Pel2)
{
  DegrainTest(TF_MID, 2, 32, 2);
}

TEST_F(TestBase, Degrain_2Blk32Pel1)
{
  DegrainTest(TF_MID, 2, 32, 1);
}

void TestBase::DegrainBinomialTest(TEST_FRAMES tf, int N, int blksize, int pel)
{
  try {
    IScriptEnvironment2* env = CreateScriptEnvironment2();

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

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
    out << "pmvb1 = s.KMPartialSuper().KMAnalyse(isb = true, delta = 2, chroma = false, blksize = " << blksize <<
      ", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false)" << std::endl;
    out << "pmvf1 = s.KMPartialSuper().KMAnalyse(isb = false, delta = 2, chroma = false, blksize = " << blksize <<
      ", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false)" << std::endl;
    out << "mvb = scuda.KMAnalyse(isb = true, delta = 1, chroma = false, blksize = " << blksize <<
      ", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false, partial = pmvb.OnCPU(0))" << std::endl;
    out << "mvf = scuda.KMAnalyse(isb = false, delta = 1, chroma = false, blksize = " << blksize <<
      ", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false, partial = pmvf.OnCPU(0))" << std::endl;
    out << "mvb1 = scuda.KMAnalyse(isb = true, delta = 2, chroma = false, blksize = " << blksize <<
      ", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false, partial = pmvb1.OnCPU(0))" << std::endl;
    out << "mvf1 = scuda.KMAnalyse(isb = false, delta = 2, chroma = false, blksize = " << blksize <<
      ", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false, partial = pmvf1.OnCPU(0))" << std::endl;

    out << "bindeg = srcuda.KMDegrain2(scuda, mvb, mvf, mvb1, mvf1, thSAD = 6400, thSCD1 = 1800, thSCD2 = 980, binomial = true).OnCUDA(0)" << std::endl;
    out << "deg1 = srcuda.KMDegrain1(scuda, mvb, mvf, thSAD = 6400, thSCD1 = 1800, thSCD2 = 980).OnCUDA(0)" << std::endl;
    out << "deg2 = srcuda.KMDegrain1(scuda, mvb1, mvf1, thSAD = 6400, thSCD1 = 1800, thSCD2 = 980).OnCUDA(0)" << std::endl;
    out << "binref = deg1.Merge(deg2, 0.2).Merge(src, 0.0625)" << std::endl;

    out << "ImageCompare(binref, bindeg, 6)" << std::endl;

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

TEST_F(TestBase, DegrainBinomial_2Blk32Pel1)
{
  DegrainBinomialTest(TF_MID, 2, 32, 1);
}

#pragma endregion

#pragma region Compensate

void TestBase::CompensateTest(TEST_FRAMES tf, int blksize, int pel)
{
  try {
    IScriptEnvironment2* env = CreateScriptEnvironment2();

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

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

TEST_F(TestBase, Compensate_Blk16Pel2)
{
  CompensateTest(TF_MID, 16, 2);
}

TEST_F(TestBase, Compensate_Blk16Pel1)
{
  CompensateTest(TF_MID, 16, 1);
}

TEST_F(TestBase, Compensate_Blk32Pel2)
{
  CompensateTest(TF_MID, 32, 2);
}

TEST_F(TestBase, Compensate_Blk32Pel1)
{
  CompensateTest(TF_MID, 32, 1);
}

#pragma endregion

#pragma region Bob

void TestBase::BobTest(TEST_FRAMES tf, bool parity)
{
  try {
    IScriptEnvironment2* env = CreateScriptEnvironment2();

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "Import(\"QTGMC_Bob.avs\")" << std::endl;

    out << "src = LWLibavVideoSource(\"test.ts\")" << (parity ? ".AssumeTFF()" : ".AssumeBFF()") << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;

    out << "ref = src.QTGMC_Bob( 0,0.5 )" << std::endl;
    out << "cuda = srcuda.KTGMC_Bob( 0,0.5 ).OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1)" << std::endl;

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

TEST_F(TestBase, BobTest_TFF)
{
  BobTest(TF_MID, true);
}

TEST_F(TestBase, BobTest_BFF)
{
  BobTest(TF_MID, false);
}

#pragma endregion

#pragma region BinomialSoften

void TestBase::BinomialSoftenTest(TEST_FRAMES tf, int radius, bool chroma)
{
  try {
    IScriptEnvironment2* env = CreateScriptEnvironment2();

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "Import(\"QTGMC_BinomialSoften.avs\")" << std::endl;

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;

    out << "ref = src.QTGMC_BinomialSoften" << radius << "(" << (chroma ? "true" : "false") << ")" << std::endl;
    out << "cuda = srcuda.KBinomialTemporalSoften(" << radius << ", 28, " << (chroma ? "true" : "false") << ").OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1" << (chroma ? "" : ", false") << ")" << std::endl;

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

TEST_F(TestBase, BinomialSoften_Rad1WithC)
{
  BinomialSoftenTest(TF_MID, 1, true);
}

TEST_F(TestBase, BinomialSoften_Rad2WithC)
{
  BinomialSoftenTest(TF_MID, 2, true);
}

TEST_F(TestBase, BinomialSoften_Rad1NoC)
{
  BinomialSoftenTest(TF_MID, 1, false);
}

TEST_F(TestBase, BinomialSoften_Rad2NoC)
{
  BinomialSoftenTest(TF_MID, 2, false);
}

#pragma endregion

#pragma region RemoveGrain

void TestBase::RemoveGrainTest(TEST_FRAMES tf, int mode, bool chroma)
{
  try {
    IScriptEnvironment2* env = CreateScriptEnvironment2();

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "Import(\"QTGMC_BinomialSoften.avs\")" << std::endl;

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;

    out << "ref = src.RemoveGrain(" << mode << (chroma ? "" : ", -1") << ")" << std::endl;
    out << "cuda = srcuda.KRemoveGrain(" << mode << (chroma ? "" : ", -1") << ").OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1" << (chroma ? "" : ", false") << ")" << std::endl;

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

TEST_F(TestBase, RemoveGrain_Mode12WithC)
{
  RemoveGrainTest(TF_MID, 12, true);
}

TEST_F(TestBase, RemoveGrain_Mode12NoC)
{
  RemoveGrainTest(TF_MID, 12, false);
}

TEST_F(TestBase, RemoveGrain_Mode20WithC)
{
  RemoveGrainTest(TF_MID, 20, true);
}

TEST_F(TestBase, RemoveGrain_Mode20NoC)
{
  RemoveGrainTest(TF_MID, 20, false);
}

#pragma endregion

#pragma region GaussResize

void TestBase::GaussResizeTest(TEST_FRAMES tf, bool chroma)
{
  try {
    IScriptEnvironment2* env = CreateScriptEnvironment2();

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;

    out << "ref = src.GaussResize(1920,1080,0,0,1920.0001,1080.0001,p=2)" << std::endl;
    out << "cuda = srcuda.KGaussResize(p=2" << (chroma ? "" : ", chroma=false") << ").OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1" << (chroma ? "" : ", false") << ")" << std::endl;

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

TEST_F(TestBase, GaussResizeTest_WithC)
{
  GaussResizeTest(TF_MID, true);
}

TEST_F(TestBase, GaussResizeTest_NoC)
{
  GaussResizeTest(TF_MID, false);
}

#pragma endregion

#pragma region InpandVerticalX2

void TestBase::InpandVerticalX2Test(TEST_FRAMES tf, bool chroma)
{
  try {
    IScriptEnvironment2* env = CreateScriptEnvironment2();

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    int rc = chroma ? 3 : 1;

    //out << "Import(\"QTGMC_BinomialSoften.avs\")" << std::endl;

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;

    out << "ref = src.mt_inpand(mode=\"vertical\", U=" << rc << ",V=" << rc << 
      ").mt_inpand(mode=\"vertical\", U=" << rc << ",V=" << rc << ")" << std::endl;
    out << "cuda = srcuda.KInpandVerticalX2(U = " << rc << ", V = " << rc << ").OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1" << (chroma ? "" : ", false") << ")" << std::endl;

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

TEST_F(TestBase, InpandVerticalX2Test_WithC)
{
  InpandVerticalX2Test(TF_MID, true);
}

TEST_F(TestBase, InpandVerticalX2Test_NoC)
{
  InpandVerticalX2Test(TF_MID, false);
}

#pragma endregion

#pragma region ExpandVerticalX2

void TestBase::ExpandVerticalX2Test(TEST_FRAMES tf, bool chroma)
{
  try {
    IScriptEnvironment2* env = CreateScriptEnvironment2();

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    int rc = chroma ? 3 : 1;

    //out << "Import(\"QTGMC_BinomialSoften.avs\")" << std::endl;

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;

    out << "ref = src.mt_expand(mode=\"vertical\", U=" << rc << ",V=" << rc <<
      ").mt_expand(mode=\"vertical\", U=" << rc << ",V=" << rc << ")" << std::endl;
    out << "cuda = srcuda.KExpandVerticalX2(U = " << rc << ", V = " << rc << ").OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1" << (chroma ? "" : ", false") << ")" << std::endl;

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

TEST_F(TestBase, ExpandVerticalX2Test_WithC)
{
  ExpandVerticalX2Test(TF_MID, true);
}

TEST_F(TestBase, ExpandVerticalX2Test_NoC)
{
  ExpandVerticalX2Test(TF_MID, false);
}

#pragma endregion

#pragma region MakeDiff

void TestBase::MakeDiffTest(TEST_FRAMES tf, bool chroma)
{
  try {
    IScriptEnvironment2* env = CreateScriptEnvironment2();

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    int rc = chroma ? 3 : 1;

    //out << "Import(\"QTGMC_BinomialSoften.avs\")" << std::endl;

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;

    out << "src2 = src.RemoveGrain(20)" << std::endl;
    out << "sr2cuda = src2.OnCPU(0)" << std::endl;

    out << "ref = src.mt_makediff(src2, U=" << rc << ",V=" << rc << ")" << std::endl;
    out << "cuda = srcuda.KMakeDiff(sr2cuda, U = " << rc << ", V = " << rc << ").OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1" << (chroma ? "" : ", false") << ")" << std::endl;

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

TEST_F(TestBase, MakeDiffTest_WithC)
{
  MakeDiffTest(TF_MID, true);
}

TEST_F(TestBase, MakeDiffTest_NoC)
{
  MakeDiffTest(TF_MID, false);
}

#pragma endregion

#pragma region Logic

void TestBase::LogicTest(TEST_FRAMES tf, const char* mode, bool chroma)
{
  try {
    IScriptEnvironment2* env = CreateScriptEnvironment2();

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    int rc = chroma ? 3 : 1;

    //out << "Import(\"QTGMC_BinomialSoften.avs\")" << std::endl;

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;

    out << "src2 = src.RemoveGrain(20)" << std::endl;
    out << "sr2cuda = src2.OnCPU(0)" << std::endl;

    out << "ref = src.mt_logic(src2, mode=\"" << mode << "\", U=" << rc << ",V=" << rc << ")" << std::endl;
    out << "cuda = srcuda.KLogic(sr2cuda, mode=\"" << mode << "\", U = " << rc << ", V = " << rc << ").OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1" << (chroma ? "" : ", false") << ")" << std::endl;

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

TEST_F(TestBase, LogicTest_MinWithC)
{
  LogicTest(TF_MID, "min", true);
}

TEST_F(TestBase, LogicTest_MinNoC)
{
  LogicTest(TF_MID, "min", false);
}

TEST_F(TestBase, LogicTest_MaxWithC)
{
  LogicTest(TF_MID, "max", true);
}

TEST_F(TestBase, LogicTest_MaxNoC)
{
  LogicTest(TF_MID, "max", false);
}

#pragma endregion

#pragma region BobShimmerFixesMerge

void TestBase::BobShimmerFixesMergeTest(TEST_FRAMES tf, int rep, bool chroma)
{
  try {
    IScriptEnvironment2* env = CreateScriptEnvironment2();

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    const char* chromastr = chroma ? "true" : "false";

    out << "Import(\"QTGMC_KeepOnlyBobShimmerFixes.avs\")" << std::endl;
    out << "Import(\"KTGMC_KeepOnlyBobShimmerFixes.avs\")" << std::endl;

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;

    out << "src2 = src.RemoveGrain(20)" << std::endl;
    out << "sr2cuda = src2.OnCPU(0)" << std::endl;

    out << "ref = src.QTGMC_KeepOnlyBobShimmerFixes(src2, " << rep << ", " << chromastr << ")" << std::endl;
    out << "cuda = srcuda.KTGMC_KeepOnlyBobShimmerFixes(sr2cuda, " << rep << ", " << chromastr << ").OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1" << (chroma ? "" : ", false") << ")" << std::endl;

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

TEST_F(TestBase, BobShimmerFixesMergeTest_Rep3WithC)
{
  BobShimmerFixesMergeTest(TF_MID, 3, true);
}

TEST_F(TestBase, BobShimmerFixesMergeTest_Rep3NoC)
{
  BobShimmerFixesMergeTest(TF_MID, 3, false);
}

TEST_F(TestBase, BobShimmerFixesMergeTest_Rep4WithC)
{
  BobShimmerFixesMergeTest(TF_MID, 4, true);
}

TEST_F(TestBase, BobShimmerFixesMergeTest_Rep4NoC)
{
  BobShimmerFixesMergeTest(TF_MID, 4, false);
}

#pragma endregion

#pragma region VResharpen

void TestBase::VResharpenTest(TEST_FRAMES tf)
{
  try {
    IScriptEnvironment2* env = CreateScriptEnvironment2();

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;

    out << "ref = Merge(mt_inpand(src,mode=\"vertical\",U=3,V=3), mt_expand(src,mode=\"vertical\",U=3,V=3))" << std::endl;
    out << "cuda = srcuda.KTGMC_VResharpen(U=3, V=3).OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1)" << std::endl;

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

TEST_F(TestBase, VResharpenTest_WithC)
{
  VResharpenTest(TF_MID);
}

#pragma endregion

#pragma region Resharpen

void TestBase::ResharpenTest(TEST_FRAMES tf)
{
  try {
    IScriptEnvironment2* env = CreateScriptEnvironment2();

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "src1 = src.RemoveGrain(20)" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;
    out << "sr1cuda = src1.OnCPU(0)" << std::endl;

    out << "ref = src.mt_lutxy(src1,\"clamp_f x x y - 0.700000 * +\",U=3,V=3)" << std::endl;
    out << "cuda = srcuda.KTGMC_Resharpen(sr1cuda, 0.70000, U=3, V=3).OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1)" << std::endl;

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

TEST_F(TestBase, ResharpenTest_WithC)
{
  ResharpenTest(TF_MID);
}

#pragma endregion

#pragma region LimitOverSharpen

void TestBase::LimitOverSharpenTest(TEST_FRAMES tf)
{
  try {
    IScriptEnvironment2* env = CreateScriptEnvironment2();

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "src1 = src.RemoveGrain(20)" << std::endl;
    out << "src2 = src1.RemoveGrain(20)" << std::endl;
    out << "src3 = src2.RemoveGrain(20)" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;
    out << "sr1cuda = src1.OnCPU(0)" << std::endl;
    out << "sr2cuda = src2.OnCPU(0)" << std::endl;
    out << "sr3cuda = src3.OnCPU(0)" << std::endl;
    
    out << "tMax = src1.mt_logic(src2, \"max\", U = 3, V = 3).mt_logic(src3, \"max\", U = 3, V = 3)" << std::endl;
    out << "tMin = src1.mt_logic(src2, \"min\", U = 3, V = 3).mt_logic(src3, \"min\", U = 3, V = 3)" << std::endl;
    out << "ref = src.mt_clamp( tMax,tMin, 0,0, U=3,V=3 )" << std::endl;

    out << "cuda = srcuda.KTGMC_LimitOverSharpen(sr1cuda, sr2cuda, sr3cuda, 0, U=3, V=3).OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1)" << std::endl;

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

TEST_F(TestBase, LimitOverSharpenTest_WithC)
{
  LimitOverSharpenTest(TF_MID);
}

#pragma endregion

#pragma region ToFullRange

void TestBase::ToFullRangeTest(TEST_FRAMES tf, bool chroma)
{
  try {
    IScriptEnvironment2* env = CreateScriptEnvironment2();

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    int rc = (chroma ? 3 : 1);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;

    out <<
      "ref = src.mt_lut(yexpr=\"0.000000 1.062500 0.066406 x 16 - 219 / 0 1 clip 0.062500 + / - * x 16 - 219 / 0 1 clip 1 0.000000 - * + 255 * \"," << 
      "expr=\"x 128 - 128 * 112 / 128 + \",y=3,u=" << rc << ",v=" << rc << ")" << std::endl;

    out << "cuda = srcuda.KTGMC_ToFullRange(u=" << rc << ",v=" << rc << ").OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1" << (chroma ? "" : ", false") << ")" << std::endl;

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

TEST_F(TestBase, ToFullRangeTest_WithC)
{
  ToFullRangeTest(TF_MID, true);
}

TEST_F(TestBase, ToFullRangeTest_NoC)
{
  ToFullRangeTest(TF_MID, false);
}

#pragma endregion

#pragma region TweakSearchClip

void TestBase::TweakSearchClipTest(TEST_FRAMES tf, bool chroma)
{
  try {
    IScriptEnvironment2* env = CreateScriptEnvironment2();

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    int rc = (chroma ? 3 : 1);

    out << "repair0 = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "bobbed = repair0.RemoveGrain(20)" << std::endl;
    out << "spatialBlur = bobbed.RemoveGrain(20)" << std::endl;
    out << "repair0cuda = repair0.OnCPU(0)" << std::endl;
    out << "bobbedcuda = bobbed.OnCPU(0)" << std::endl;
    out << "spatialBlurcuda = spatialBlur.OnCPU(0)" << std::endl;

    out << "tweaked = mt_lutxy( repair0, bobbed, \"x 3 scalef + y < x 3 scalef + x 3 scalef - y > x 3 scalef - y ? ?\", u=" << rc << ",v=" << rc << ")" << std::endl;
    out << "ref = spatialBlur.mt_lutxy( tweaked, \"x 7 scalef + y < x 2 scalef + x 7 scalef - y > x 2 scalef - x 51 * y 49 * + 100 / ? ?\", u=" << rc << ",v=" << rc << ")" << std::endl;

    out << "cuda = repair0cuda.KTGMC_TweakSearchClip(bobbedcuda, spatialBlurcuda, u=" << rc << ",v=" << rc << ").OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1" << (chroma ? "" : ", false") << ")" << std::endl;

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

TEST_F(TestBase, TweakSearchClipTest_WithC)
{
  TweakSearchClipTest(TF_MID, true);
}

TEST_F(TestBase, TweakSearchClipTest_NoC)
{
  TweakSearchClipTest(TF_MID, false);
}

#pragma endregion

#pragma region Merge

void TestBase::MergeTest(TEST_FRAMES tf, bool chroma)
{
  try {
    IScriptEnvironment2* env = CreateScriptEnvironment2();

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "src1 = src.RemoveGrain(20)" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;
    out << "sr1cuda = src1.OnCPU(0)" << std::endl;

    if (chroma) {
      out << "ref = src.Merge(src1, 0.844)" << std::endl;
      out << "cuda = srcuda.KMerge(sr1cuda, 0.844).OnCUDA(0)" << std::endl;
    }
    else {
      out << "ref = src.MergeLuma(src1, 0.844)" << std::endl;
      out << "cuda = srcuda.KMergeLuma(sr1cuda, 0.844).OnCUDA(0)" << std::endl;
    }

    out << "ImageCompare(ref, cuda, 1" << (chroma ? "" : ", false") << ")" << std::endl;

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

TEST_F(TestBase, MergeTest_WithC)
{
  MergeTest(TF_MID, true);
}

TEST_F(TestBase, MergeTest_NoC)
{
  MergeTest(TF_MID, false);
}

#pragma endregion

#pragma region NNEDI3

void TestBase::NNEDI3Test(TEST_FRAMES tf, bool chroma)
{
  try {
    IScriptEnvironment2* env = CreateScriptEnvironment2();;

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KNNEDI3.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "src1 = src.RemoveGrain(20)" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;
    out << "sr1cuda = src1.OnCPU(0)" << std::endl;

    out << "ref = src.KNNEDI3(clip18,field=-2,nsize=1,nns=1,qual=1,threads=0,U=True,V=True)" << std::endl;
    out << "cuda = srcuda.KTGMC_Resharpen(sr1cuda, 0.70000, U=3, V=3).OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1)" << std::endl;

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

TEST_F(TestBase, DISABLED_NNEDI3Test_WithC)
{
  NNEDI3Test(TF_MID, true);
}

TEST_F(TestBase, DISABLED_NNEDI3Test_NoC)
{
  NNEDI3Test(TF_MID, false);
}

#pragma endregion

int main(int argc, char **argv)
{
	::testing::GTEST_FLAG(filter) = "TestBase.*";
	::testing::InitGoogleTest(&argc, argv);
	int result = RUN_ALL_TESTS();

	//getchar();

	return result;
}

