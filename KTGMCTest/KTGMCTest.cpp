
#define _CRT_SECURE_NO_WARNINGS

#include <Windows.h>

#define AVS_LINKAGE_DLLIMPORT
#include "avisynth.h"
#pragma comment(lib, "avisynth.lib")

#include "gtest/gtest.h"

#include <fstream>
#include <string>
#include <iostream>
#include <memory>

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

struct ScriptEnvironmentDeleter {
  void operator()(IScriptEnvironment* env) {
    env->DeleteScriptEnvironment();
  }
};

typedef std::unique_ptr<IScriptEnvironment2, ScriptEnvironmentDeleter> PEnv;

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

  void MSuperTest(TEST_FRAMES tf, bool chroma, int pel, int level);
  void AnalyzeTest(TEST_FRAMES tf, bool cuda, int blksize, bool chroma, int pel, int batch);
  void DegrainTest(TEST_FRAMES tf, int N, int blksize, int pel);
  void DegrainBinomialTest(TEST_FRAMES tf, int N, int blksize, int pel);
  void CompensateTest(TEST_FRAMES tf, int blksize, int pel);
  void MVReplaceTest(TEST_FRAMES tf, bool kvm);

  void BobTest(TEST_FRAMES tf, bool parity);
  void BinomialSoftenTest(TEST_FRAMES tf, int radius, bool chroma);
  void RemoveGrainTest(TEST_FRAMES tf, int mode, bool chroma);
  void RepairTest(TEST_FRAMES tf, int mode, bool chroma);
  void VerticalCleanerTest(TEST_FRAMES tf, int mode, bool chroma);
  void GaussResizeTest(TEST_FRAMES tf, bool chroma);

  void InpandVerticalX2Test(TEST_FRAMES tf, bool chroma);
  void ExpandVerticalX2Test(TEST_FRAMES tf, bool chroma);
  void MakeDiffTest(TEST_FRAMES tf, bool chroma, bool makediff);
  void LogicTest(TEST_FRAMES tf, const char* mode, bool chroma);

  void BobShimmerFixesMergeTest(TEST_FRAMES tf, int rep, bool chroma);
  void VResharpenTest(TEST_FRAMES tf);
  void ResharpenTest(TEST_FRAMES tf);
  void LimitOverSharpenTest(TEST_FRAMES tf);
  void ToFullRangeTest(TEST_FRAMES tf, bool chroma);
  void TweakSearchClipTest(TEST_FRAMES tf, bool chroma);
  void LosslessProcTest(TEST_FRAMES tf, bool chroma);
  void ErrorAdjustTest(TEST_FRAMES tf, bool chroma);

  void MergeTest(TEST_FRAMES tf, bool chroma);
  void WeaveTest(TEST_FRAMES tf, bool parity, bool dbl);
  void CopyTest(TEST_FRAMES tf, bool cuda);

  void NNEDI3Test(TEST_FRAMES tf, bool chroma, int nsize, int nns, int qual, int pscrn);
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
    clip->GetFrame(2, env);
    clip->GetFrame(3, env);
    clip->GetFrame(4, env);
    clip->GetFrame(5, env);
    break;
  case TF_END:
    clip->GetFrame(nframes - 6, env);
    clip->GetFrame(nframes - 5, env);
    clip->GetFrame(nframes - 4, env);
    clip->GetFrame(nframes - 3, env);
    clip->GetFrame(nframes - 2, env);
    clip->GetFrame(nframes - 1, env);
    break;
  }
}

#pragma region MSuper

void TestBase::MSuperTest(TEST_FRAMES tf, bool chroma, int pel, int level)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    const char* chromastr = chroma ? "true" : "false";

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "ref = src.MSuper(chroma = " << chromastr << ", pel = " << pel << ", levels = " << level << ")" << std::endl;
    out << "cuda = src.OnCPU(0).KMSuper(chroma = " << chromastr << ", pel = " << pel << ", levels = " << level << ").OnCUDA(0)" << std::endl;
    out << "KMSuperCheck(cuda, ref, src)" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      GetFrames(clip, tf, env.get());
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

TEST_F(TestBase, MSuper_WithCPel1Level0)
{
  MSuperTest(TF_MID, true, 1, 0);
}

TEST_F(TestBase, MSuper_WithCPel2Level0)
{
  MSuperTest(TF_MID, true, 2, 0);
}

TEST_F(TestBase, MSuper_WithCPel1Level1)
{
  MSuperTest(TF_MID, true, 1, 1);
}

TEST_F(TestBase, MSuper_WithCPel2Level1)
{
  MSuperTest(TF_MID, true, 2, 1);
}

TEST_F(TestBase, MSuper_NoCPel1Level0)
{
  MSuperTest(TF_MID, false, 1, 0);
}

#pragma endregion

#pragma region Analyze

void TestBase::AnalyzeTest(TEST_FRAMES tf, bool cuda, int blksize, bool chroma, int pel, int batch)
{
  PEnv env;
	try {
    env = PEnv(CreateScriptEnvironment2());

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
      GetFrames(clip, tf, env.get());
		}
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

TEST_F(TestBase, Analyze_Blk32NoCPel1Begin)
{
  AnalyzeTest(TF_BEGIN, true, 32, false, 1, 1);
}

TEST_F(TestBase, Analyze_Blk32NoCPel1End)
{
  AnalyzeTest(TF_END, true, 32, false, 1, 1);
}

TEST_F(TestBase, Analyze_Blk32NoCPel1Batch8End)
{
  AnalyzeTest(TF_END, true, 32, false, 1, 8);
}

#pragma endregion

#pragma region Degrain

void TestBase::DegrainTest(TEST_FRAMES tf, int N, int blksize, int pel)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

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
      GetFrames(clip, tf, env.get());
    }
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

TEST_F(TestBase, Degrain_2Blk16Pel2Begin)
{
  DegrainTest(TF_BEGIN, 2, 16, 2);
}

TEST_F(TestBase, Degrain_2Blk16Pel1End)
{
  DegrainTest(TF_END, 2, 16, 2);
}

void TestBase::DegrainBinomialTest(TEST_FRAMES tf, int N, int blksize, int pel)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

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
      GetFrames(clip, tf, env.get());
    }
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
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

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
      GetFrames(clip, tf, env.get());
    }
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

TEST_F(TestBase, Compensate_Blk16Pel2Begin)
{
  CompensateTest(TF_BEGIN, 16, 2);
}

TEST_F(TestBase, Compensate_Blk16Pel2End)
{
  CompensateTest(TF_END, 16, 2);
}

#pragma endregion

#pragma region MVReplace

void TestBase::MVReplaceTest(TEST_FRAMES tf, bool kvm)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;

    out << "ks = src.KMSuper(pel = 2)" << std::endl;
    out << "kmvb = ks.KMAnalyse(isb = true, delta = 1, chroma = false, blksize = 32, overlap = 16, lambda = 400, global = true, meander = false)" << std::endl;
    out << "ms = src.MSuper(pel = 2)" << std::endl;
    out << "mmvb = ms.MAnalyse(isb = true, delta = 1, chroma = false, blksize = 32, overlap = 16, lambda = 400, global = true, meander = false)" << std::endl;

    if (kvm) {
      out << "kcom = src.KMCompensate(ks, kmvb,thSCD1=1800,thSCD2=980)" << std::endl;
      out << "mcom = src.MCompensate(ms, MVReplaceWithKMV(mmvb, kmvb),thSCD1=1800,thSCD2=980)" << std::endl;
    }
    else {
      out << "kcom = src.KMCompensate(ks, KMVReplaceWithMV(kmvb, mmvb),thSCD1=1800,thSCD2=980)" << std::endl;
      out << "mcom = src.MCompensate(ms, mmvb,thSCD1=1800,thSCD2=980)" << std::endl;
    }

    out << "ImageCompare(kcom, mcom, 1)" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      GetFrames(clip, tf, env.get());
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

TEST_F(TestBase, MVReplace_KMV)
{
  MVReplaceTest(TF_MID, true);
}

TEST_F(TestBase, MVReplace_MV)
{
  MVReplaceTest(TF_MID, false);
}

#pragma endregion

#pragma region Bob

void TestBase::BobTest(TEST_FRAMES tf, bool parity)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

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
      GetFrames(clip, tf, env.get());
    }
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
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

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
      GetFrames(clip, tf, env.get());
    }
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
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;

    out << "ref = src.RemoveGrain(" << mode << (chroma ? "" : ", -1") << ")" << std::endl;
    out << "cuda = srcuda.KRemoveGrain(" << mode << (chroma ? "" : ", -1") << ").OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1" << (chroma ? "" : ", false") << ")" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      GetFrames(clip, tf, env.get());
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

TEST_F(TestBase, RemoveGrain_Mode1WithC)
{
  RemoveGrainTest(TF_MID, 1, true);
}

TEST_F(TestBase, RemoveGrain_Mode1NoC)
{
  RemoveGrainTest(TF_MID, 1, false);
}

TEST_F(TestBase, RemoveGrain_Mode2WithC)
{
  RemoveGrainTest(TF_MID, 2, true);
}

TEST_F(TestBase, RemoveGrain_Mode2NoC)
{
  RemoveGrainTest(TF_MID, 2, false);
}

TEST_F(TestBase, RemoveGrain_Mode3WithC)
{
  RemoveGrainTest(TF_MID, 3, true);
}

TEST_F(TestBase, RemoveGrain_Mode3NoC)
{
  RemoveGrainTest(TF_MID, 3, false);
}

TEST_F(TestBase, RemoveGrain_Mode4WithC)
{
  RemoveGrainTest(TF_MID, 4, true);
}

TEST_F(TestBase, RemoveGrain_Mode4NoC)
{
  RemoveGrainTest(TF_MID, 4, false);
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

#pragma region Repair

void TestBase::RepairTest(TEST_FRAMES tf, int mode, bool chroma)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "sref = src.GaussResize(1920,1080,0,0,1920.0001,1080.0001,p=2)" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;
    out << "srefcuda = sref.OnCPU(0)" << std::endl;

    out << "ref = src.Repair(sref, " << mode << (chroma ? "" : ", -1") << ")" << std::endl;
    out << "cuda = srcuda.KRepair(srefcuda, " << mode << (chroma ? "" : ", -1") << ").OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1" << (chroma ? "" : ", false") << ")" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      GetFrames(clip, tf, env.get());
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

TEST_F(TestBase, Repair_Mode1WithC)
{
  RepairTest(TF_MID, 1, true);
}

TEST_F(TestBase, Repair_Mode1NoC)
{
  RepairTest(TF_MID, 1, false);
}

TEST_F(TestBase, Repair_Mode2WithC)
{
  RepairTest(TF_MID, 2, true);
}

TEST_F(TestBase, Repair_Mode2NoC)
{
  RepairTest(TF_MID, 2, false);
}

TEST_F(TestBase, Repair_Mode3WithC)
{
  RepairTest(TF_MID, 3, true);
}

TEST_F(TestBase, Repair_Mode3NoC)
{
  RepairTest(TF_MID, 3, false);
}

TEST_F(TestBase, Repair_Mode4WithC)
{
  RepairTest(TF_MID, 4, true);
}

TEST_F(TestBase, Repair_Mode4NoC)
{
  RepairTest(TF_MID, 4, false);
}

#pragma endregion

#pragma region VerticalCleaner

void TestBase::VerticalCleanerTest(TEST_FRAMES tf, int mode, bool chroma)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;

    out << "ref = src.VerticalCleaner(" << mode << (chroma ? "" : ", 0") << ")" << std::endl;
    out << "cuda = srcuda.KVerticalCleaner(" << mode << (chroma ? "" : ", 0") << ").OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1)" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      GetFrames(clip, tf, env.get());
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

TEST_F(TestBase, VerticalCleaner_WithC)
{
  VerticalCleanerTest(TF_MID, 1, true);
}

TEST_F(TestBase, VerticalCleaner_NoC)
{
  VerticalCleanerTest(TF_MID, 1, false);
}

#pragma endregion

#pragma region GaussResize

void TestBase::GaussResizeTest(TEST_FRAMES tf, bool chroma)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

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
      GetFrames(clip, tf, env.get());
    }
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
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

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
      GetFrames(clip, tf, env.get());
    }
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
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

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
      GetFrames(clip, tf, env.get());
    }
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

void TestBase::MakeDiffTest(TEST_FRAMES tf, bool chroma, bool makediff)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

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

    if (makediff) {
      out << "ref = src.mt_makediff(src2, U=" << rc << ",V=" << rc << ")" << std::endl;
      out << "cuda = srcuda.KMakeDiff(sr2cuda, U = " << rc << ", V = " << rc << ").OnCUDA(0)" << std::endl;
    }
    else {
      out << "ref = src.mt_adddiff(src2, U=" << rc << ",V=" << rc << ")" << std::endl;
      out << "cuda = srcuda.KAddDiff(sr2cuda, U = " << rc << ", V = " << rc << ").OnCUDA(0)" << std::endl;
    }

    out << "ImageCompare(ref, cuda, 1" << (chroma ? "" : ", false") << ")" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      GetFrames(clip, tf, env.get());
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

TEST_F(TestBase, MakeDiffTest_WithC)
{
  MakeDiffTest(TF_MID, true, true);
}

TEST_F(TestBase, MakeDiffTest_NoC)
{
  MakeDiffTest(TF_MID, false, true);
}

TEST_F(TestBase, AddDiffTest_WithC)
{
  MakeDiffTest(TF_MID, true, false);
}

TEST_F(TestBase, AddDiffTest_NoC)
{
  MakeDiffTest(TF_MID, false, false);
}

#pragma endregion

#pragma region Logic

void TestBase::LogicTest(TEST_FRAMES tf, const char* mode, bool chroma)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

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
      GetFrames(clip, tf, env.get());
    }
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
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

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
      GetFrames(clip, tf, env.get());
    }
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
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

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
      GetFrames(clip, tf, env.get());
    }
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
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "src1 = src.RemoveGrain(20).RemoveGrain(20)" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;
    out << "sr1cuda = src1.OnCPU(0)" << std::endl;

    out << "ref = src.mt_lutxy(src1,\"clamp_f x x y - 0.700000 * +\",U=3,V=3)" << std::endl;
    out << "cuda = srcuda.KTGMC_Resharpen(sr1cuda, 0.70000, U=3, V=3).OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1)" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      GetFrames(clip, tf, env.get());
    }
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
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

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
      GetFrames(clip, tf, env.get());
    }
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
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

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
      GetFrames(clip, tf, env.get());
    }
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
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

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
      GetFrames(clip, tf, env.get());
    }
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

#pragma region LosslessProc

void TestBase::LosslessProcTest(TEST_FRAMES tf, bool chroma)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    int rc = (chroma ? 3 : 1);

    out << "x = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "y = x.RemoveGrain(20)" << std::endl;
    out << "xcuda = x.OnCPU(0)" << std::endl;
    out << "ycuda = y.OnCPU(0)" << std::endl;

    out << "ref = mt_lutxy(x, y,\"x range_half - y range_half - * 0 < range_half x range_half - abs y range_half - abs < x y ? ?\",u=" << rc << ",v=" << rc << ")" << std::endl;
    out << "cuda = KTGMC_LosslessProc(xcuda, ycuda, u=" << rc << ",v=" << rc << ").OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1" << (chroma ? "" : ", false") << ")" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      GetFrames(clip, tf, env.get());
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

TEST_F(TestBase, LosslessProcTest_WithC)
{
  LosslessProcTest(TF_MID, true);
}

TEST_F(TestBase, LosslessProcTest_NoC)
{
  LosslessProcTest(TF_MID, false);
}

#pragma endregion

#pragma region Merge

void TestBase::MergeTest(TEST_FRAMES tf, bool chroma)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

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
      GetFrames(clip, tf, env.get());
    }
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

#pragma region Weave

void TestBase::WeaveTest(TEST_FRAMES tf, bool parity, bool dbl)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    if (parity) {
      out << "src.AssumeTFF()" << std::endl;
    }
    else {
      out << "src.AssumeBFF()" << std::endl;
    }
    out << "srcuda = src.OnCPU(0)" << std::endl;

    if (dbl) {
      out << "ref = src.SeparateFields().SelectEvery( 4, 0,3 ).DoubleWeave()" << std::endl;
      out << "cuda = srcuda.SeparateFields().SelectEvery( 4, 0,3 ).KDoubleWeave().OnCUDA(0)" << std::endl;
    }
    else {
      out << "ref = src.SeparateFields().SelectEvery( 4, 0,3 ).Weave()" << std::endl;
      out << "cuda = srcuda.SeparateFields().SelectEvery( 4, 0,3 ).KWeave().OnCUDA(0)" << std::endl;
    }

    out << "ImageCompare(ref, cuda, 0)" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      GetFrames(clip, tf, env.get());
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

TEST_F(TestBase, WeaveTest_DoubleTFF)
{
  WeaveTest(TF_MID, true, true);
}

TEST_F(TestBase, WeaveTest_DoubleBFF)
{
  WeaveTest(TF_MID, false, true);
}

TEST_F(TestBase, WeaveTest_TFF)
{
  WeaveTest(TF_MID, true, false);
}

TEST_F(TestBase, WeaveTest_BFF)
{
  WeaveTest(TF_MID, false, false);
}

#pragma endregion

#pragma region Copy

void TestBase::CopyTest(TEST_FRAMES tf, bool cuda)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;

    if (cuda) {
      out << "srcuda = src.OnCPU(0)" << std::endl;
      out << "ref = src.SeparateFields().SelectEvery( 4, 0,3 )" << std::endl;
      out << "cuda = srcuda.SeparateFields().SelectEvery( 4, 0,3 ).KCopy().OnCUDA(0)" << std::endl;
    }
    else {
      out << "ref = src.SeparateFields().SelectEvery( 4, 0,3 )" << std::endl;
      out << "cuda = ref.KCopy()" << std::endl;
    }

    out << "ImageCompare(ref, cuda, 0)" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      GetFrames(clip, TF_MID, env.get());
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

TEST_F(TestBase, CopyTest_CUDA)
{
  CopyTest(TF_MID, true);
}

TEST_F(TestBase, CopyTest_CPU)
{
  CopyTest(TF_MID, false);
}

#pragma endregion

#pragma region ErrorAdjust

void TestBase::ErrorAdjustTest(TEST_FRAMES tf, bool chroma)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    float errorAdj = 1.333333f;
    int rc = chroma ? 3 : 1;

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;

    out << "src2 = src.RemoveGrain(20)" << std::endl;
    out << "sr2cuda = src2.OnCPU(0)" << std::endl;

    out << "ref = src.mt_lutxy(src2, \"clamp_f x " << (errorAdj + 1) << " * y " << errorAdj << " * -\", U=" << rc << ", V = " << rc << " )" << std::endl;
    out << "cuda = srcuda.KTGMC_ErrorAdjust(sr2cuda, " << errorAdj << ", U=" << rc << ", V = " << rc << ").OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1)" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      GetFrames(clip, tf, env.get());
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

TEST_F(TestBase, ErrorAdjustTest_WithC)
{
  ErrorAdjustTest(TF_MID, true);
}

#pragma endregion

#pragma region NNEDI3

void TestBase::NNEDI3Test(TEST_FRAMES tf, bool chroma, int nsize, int nns, int qual, int pscrn)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KNNEDI3.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    const char* UV = (chroma ? "True" : "False");

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;

    // opt>1だと一致しなくなるのでopt=1（SIMDなし）を指定
    out <<
      "ref = src.KNNEDI3(field=-2,nsize=" << nsize << ",nns=" << nns << ",qual=" << qual << "," <<
      "pscrn=" << pscrn << ",opt=1,U=" << UV << ",V=" << UV << ")" << std::endl;
    out <<
      "cuda = srcuda.KNNEDI3(field=-2,nsize=" << nsize << ",nns=" << nns << ",qual=" << qual << "," <<
      "pscrn=" << pscrn << ",U=" << UV << ",V=" << UV << ").OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1" << (chroma ? "" : ", false") << ")" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      GetFrames(clip, tf, env.get());
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

TEST_F(TestBase, NNEDI3Test_NS0NN0Q1PS2)
{
  NNEDI3Test(TF_MID, true, 0, 0, 1, 2);
}

TEST_F(TestBase, NNEDI3Test_NS1NN0Q1PS2)
{
  NNEDI3Test(TF_MID, true, 1, 0, 1, 2);
}

TEST_F(TestBase, NNEDI3Test_NS2NN0Q1PS2)
{
  NNEDI3Test(TF_MID, true, 2, 0, 1, 2);
}

TEST_F(TestBase, NNEDI3Test_NS3NN0Q1PS2)
{
  NNEDI3Test(TF_MID, true, 3, 0, 1, 2);
}

TEST_F(TestBase, NNEDI3Test_NS4NN0Q1PS2)
{
  NNEDI3Test(TF_MID, true, 4, 0, 1, 2);
}

TEST_F(TestBase, NNEDI3Test_NS5NN0Q1PS2)
{
  NNEDI3Test(TF_MID, true, 5, 0, 1, 2);
}

TEST_F(TestBase, NNEDI3Test_NS6NN0Q1PS2)
{
  NNEDI3Test(TF_MID, true, 6, 0, 1, 2);
}

TEST_F(TestBase, NNEDI3Test_NS0NN1Q1PS2)
{
  NNEDI3Test(TF_MID, true, 0, 1, 1, 2);
}

TEST_F(TestBase, NNEDI3Test_NS0NN2Q1PS2)
{
  NNEDI3Test(TF_MID, true, 0, 2, 1, 2);
}

TEST_F(TestBase, NNEDI3Test_NS0NN3Q1PS2)
{
  NNEDI3Test(TF_MID, true, 0, 3, 1, 2);
}

TEST_F(TestBase, NNEDI3Test_NS0NN4Q1PS2)
{
  NNEDI3Test(TF_MID, true, 0, 4, 1, 2);
}

TEST_F(TestBase, NNEDI3Test_NS0NN0Q2PS2)
{
  NNEDI3Test(TF_MID, true, 0, 0, 2, 2);
}

// 性能評価用
TEST_F(TestBase, NNEDI3Test_NS1NN1Q1PS2)
{
  NNEDI3Test(TF_MID, true, 1, 1, 1, 2);
}

TEST_F(TestBase, NNEDI3Test_NoC)
{
  NNEDI3Test(TF_MID, false, 0, 0, 1, 2);
}

// 性能評価用
TEST_F(TestBase, NNEDI3Test_Perf)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KNNEDI3.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    const char* UV = "True";

    out << "SetLogParams(\"avsrun.log\", LOG_WARNING)" << std::endl;
    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
#if 1
    out << "return src.OnCPU(1)" << std::endl;
#else
    out << "srcuda = src.OnCPU(2)" << std::endl;
    out << "return srcuda.KNNEDI3(field=-2,nsize=1,nns=1,qual=1,pscrn=2,opt=1,threads=1,U=" << UV << ",V=" << UV << ").OnCUDA(2)" << std::endl;
#endif

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      for (int i = 100; i < 1100; ++i) {
        clip->GetFrame(i, env.get());
      }
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

#pragma endregion

#pragma region AviSynthPlus

TEST_F(TestBase, DeviceCheck)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);
    out << "LWLibavVideoSource(\"test.ts\").OnCUDA(0)" << std::endl;
    out.close();

    EXPECT_THROW(env->Invoke("Import", scriptpath.c_str()).AsClip(), AvisynthError);
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

TEST_F(TestBase, StartupTime)
{
	PEnv env;
	try {
		env = PEnv(CreateScriptEnvironment2());

		AVSValue result;
		std::string scriptpath = workDirPath + "\\script.avs";

		std::ofstream out(scriptpath);
		out << "SetCacheMode(CACHE_OPTIMAL_SIZE)" << std::endl;
		out << "LWLibavVideoSource(\"test.ts\").QTGMC()" << std::endl;
		out.close();

		PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();

		int64_t sum = 0, prev, cur;
		for (int i = 100; i < 120; ++i) {
			QueryPerformanceCounter((LARGE_INTEGER*)&prev);

			PVideoFrame frame = clip->GetFrame(i, env.get());

			QueryPerformanceCounter((LARGE_INTEGER*)&cur);
			int64_t frametime = cur - prev;
			sum += frametime;

			int64_t freq;
			QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
			printf("%d: %.1f (elapsed: %.1f)\n", i - 100 + 1,
				(double)frametime / freq * 1000.0, (double)sum / freq * 1000.0);
		}
	}
	catch (const AvisynthError& err) {
		printf("%s\n", err.msg);
		GTEST_FAIL();
	}
}

#pragma endregion

// 性能評価用
TEST_F(TestBase, KTGMC_Perf)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string ktgmcPath = modulePath + "\\KTGMC.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);
    std::string knnediPath = modulePath + "\\KNNEDI3.dll";
    env->LoadPlugin(knnediPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "SetLogParams(\"avsrun.log\", LOG_WARNING)" << std::endl;
    out << "SetMemoryMax(2048, type=DEV_TYPE_CUDA)" << std::endl;
    out << "Import(\"KTGMC.avsi\")" << std::endl;
    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "src.OnCPU(2).KTGMC().OnCUDA(2)" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      for (int i = 100; i < 104; ++i) {
        clip->GetFrame(i, env.get());
        printf("===== FRAME %d COMPLETE =====\n", i);
      }
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

TEST_F(TestBase, MemoryLeak)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string scriptpath = workDirPath + "\\script.avs";
    std::ofstream out(scriptpath);

    out << "SetLogParams(\"avsrun.log\", LOG_WARNING)" << std::endl;
    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "return src.OnCPU(1).OnCUDA(0)" << std::endl;
    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      for (int i = 100; i < 101; ++i) {
        clip->GetFrame(i, env.get());
      }
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

TEST_F(TestBase, DeviceMatchingBug)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string scriptpath = workDirPath + "\\script.avs";
    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "src.OnCPU(0).OnCUDA(0).AudioDub(LWLibavAudioSource(\"test.ts\"))" << std::endl;
    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      for (int i = 100; i < 101; ++i) {
        clip->GetFrame(i, env.get());
      }
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

#pragma region MergeStatic

TEST_F(TestBase, AnalyzeStaticTest)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KFM.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;

    out << "ref = src.KAnalyzeStatic(30, 15)" << std::endl;
    out << "cuda = srcuda.KAnalyzeStatic(30, 15).OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1)" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      GetFrames(clip, TF_MID, env.get());
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

TEST_F(TestBase, MergeStaticTest)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KFM.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "bb = src.Bob()" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;
    out << "bbcuda = bb.OnCPU(0)" << std::endl;

    out << "stt = src.KAnalyzeStatic(30, 15)" << std::endl;
    out << "ref = bb.KMergeStatic(src, stt)" << std::endl;
    out << "cuda = bbcuda.KMergeStatic(srcuda, stt.OnCPU(0)).OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1)" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      GetFrames(clip, TF_MID, env.get());
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

#pragma endregion

#pragma region Telecine

TEST_F(TestBase, AnalyzeFrameTest)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KFM.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;

    out << "ref = src.KFMFrameAnalyze()" << std::endl;
    out << "cuda = srcuda.KFMFrameAnalyze().OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1)" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      GetFrames(clip, TF_MID, env.get());
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

TEST_F(TestBase, TelecineTest)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KFM.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;

    out << "fm = src.KFMFrameAnalyze(15, 7, 20, 8).KFMCycleAnalyze(src)" << std::endl;
    out << "ref = src.KTelecine(fm)" << std::endl;
    out << "cuda = srcuda.KTelecine(fm.OnCPU(0)).OnCUDA(0)" << std::endl;

    out << "ImageCompare(ref, cuda, 1)" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      GetFrames(clip, TF_MID, env.get());
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

TEST_F(TestBase, RemoveCombeTest)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\KFM.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "srcuda = src.OnCPU(0)" << std::endl;

    out << "ref = src.KRemoveCombe(30, 50, 150, 0, 5)" << std::endl;
    out << "cuda = srcuda.KRemoveCombe(30, 50, 150, 0, 5).OnCUDA(0)" << std::endl;

    out << "check = KRemoveCombeCheck(ref, cuda)" << std::endl;
    out << "ImageCompare(ref, cuda, 1)" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      GetFrames(clip, TF_MID, env.get());
      PClip check = env->GetVar("check").AsClip();
      GetFrames(check, TF_MID, env.get());
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

#pragma endregion

int main(int argc, char **argv)
{
	::testing::GTEST_FLAG(filter) = "TestBase.StartupTime*";
	::testing::InitGoogleTest(&argc, argv);
	int result = RUN_ALL_TESTS();

	//getchar();

	return result;
}

