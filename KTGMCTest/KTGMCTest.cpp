
#include "TestCommons.h"

// テスト対象となるクラス Foo のためのフィクスチャ
class KTGMCTest : public AvsTestBase {
protected:
	KTGMCTest() { }

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

	void TemporalNRTest(TEST_FRAMES tf);
	void DebandTest(TEST_FRAMES tf, int sample_mode, bool blur_first);
	void EdgeLevelTest(TEST_FRAMES tf, int repair, bool chroma);

   void CFieldDiffTest(int nt, bool chroma);
   void CFrameDiffDupTest(int blocksize, bool chroma);
};

#pragma region MSuper

void KTGMCTest::MSuperTest(TEST_FRAMES tf, bool chroma, int pel, int level)
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
    out << "cuda = src.OnCPU(0).KMSuper(chroma = " << chromastr << ", pel = " << pel << ", levels = " << level << ")" O_C(0) << std::endl;
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

TEST_F(KTGMCTest, MSuper_WithCPel1Level0)
{
  MSuperTest(TF_MID, true, 1, 0);
}

TEST_F(KTGMCTest, MSuper_WithCPel2Level0)
{
  MSuperTest(TF_MID, true, 2, 0);
}

TEST_F(KTGMCTest, MSuper_WithCPel1Level1)
{
  MSuperTest(TF_MID, true, 1, 1);
}

TEST_F(KTGMCTest, MSuper_WithCPel2Level1)
{
  MSuperTest(TF_MID, true, 2, 1);
}

TEST_F(KTGMCTest, MSuper_NoCPel1Level0)
{
  MSuperTest(TF_MID, false, 1, 0);
}

#pragma endregion

#pragma region Analyze

void KTGMCTest::AnalyzeTest(TEST_FRAMES tf, bool cuda, int blksize, bool chroma, int pel, int batch)
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
      (cuda ? ".OnCPU(0)" : "") << ")" << (cuda ? O_C(0) : "") << std::endl;
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

TEST_F(KTGMCTest, Analyze_Blk32NoCPel1Batch1)
{
  AnalyzeTest(TF_MID, true, 32, false, 1, 1);
}

TEST_F(KTGMCTest, Analyze_Blk32NoCPel1Batch2)
{
  AnalyzeTest(TF_MID, true, 32, false, 1, 2);
}

TEST_F(KTGMCTest, Analyze_Blk32NoCPel1Batch3)
{
  AnalyzeTest(TF_MID, true, 32, false, 1, 3);
}

TEST_F(KTGMCTest, Analyze_Blk32NoCPel1Batch8)
{
  AnalyzeTest(TF_MID, true, 32, false, 1, 8);
}

TEST_F(KTGMCTest, Analyze_Blk16WithCPel2)
{
	AnalyzeTest(TF_MID, true, 16, true, 2, 4);
}

TEST_F(KTGMCTest, Analyze_Blk16WithCPel1)
{
	AnalyzeTest(TF_MID, true, 16, true, 1, 4);
}

TEST_F(KTGMCTest, Analyze_Blk16NoCPel2)
{
	AnalyzeTest(TF_MID, true, 16, false, 2, 4);
}

TEST_F(KTGMCTest, Analyze_Blk16NoCPel1)
{
	AnalyzeTest(TF_MID, true, 16, false, 1, 4);
}

TEST_F(KTGMCTest, Analyze_Blk8WithCPel2)
{
	AnalyzeTest(TF_MID, true, 8, true, 2, 4);
}

TEST_F(KTGMCTest, Analyze_Blk8WithCPel1)
{
	AnalyzeTest(TF_MID, true, 8, true, 1, 4);
}

TEST_F(KTGMCTest, Analyze_Blk8NoCPel2)
{
	AnalyzeTest(TF_MID, true, 8, false, 2, 4);
}

TEST_F(KTGMCTest, Analyze_Blk8NoCPel1)
{
	AnalyzeTest(TF_MID, true, 8, false, 1, 4);
}

TEST_F(KTGMCTest, Analyze_Blk32WithCPel2)
{
	AnalyzeTest(TF_MID, true, 32, true, 2, 4);
}

TEST_F(KTGMCTest, Analyze_Blk32WithCPel1)
{
	AnalyzeTest(TF_MID, true, 32, true, 1, 4);
}

TEST_F(KTGMCTest, Analyze_Blk32NoCPel2)
{
	AnalyzeTest(TF_MID, true, 32, false, 2, 4);
}

TEST_F(KTGMCTest, Analyze_Blk32NoCPel1)
{
	AnalyzeTest(TF_MID, true, 32, false, 1, 4);
}

TEST_F(KTGMCTest, AnalyzeCPU_Blk16WithCPel2)
{
  AnalyzeTest(TF_MID, false, 16, true, 2, 4);
}

TEST_F(KTGMCTest, Analyze_Blk32NoCPel1Begin)
{
  AnalyzeTest(TF_BEGIN, true, 32, false, 1, 1);
}

TEST_F(KTGMCTest, Analyze_Blk32NoCPel1End)
{
  AnalyzeTest(TF_END, true, 32, false, 1, 1);
}

TEST_F(KTGMCTest, Analyze_Blk32NoCPel1Batch8End)
{
  AnalyzeTest(TF_END, true, 32, false, 1, 8);
}

#pragma endregion

#pragma region Degrain

void KTGMCTest::DegrainTest(TEST_FRAMES tf, int N, int blksize, int pel)
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
				out << "degref = src.KMDegrain1(s, mvb"<< O_C(0) <<", mvf"<< O_C(0)<<", thSAD = 6400, thSCD1 = 1800, thSCD2 = 980)" << std::endl;
        out << "degcuda = srcuda.KMDegrain1(scuda, mvb, mvf, thSAD = 6400, thSCD1 = 1800, thSCD2 = 980)"<< O_C(0) << std::endl;
      }
      else if (N == 2) {
        out << "degref = src.KMDegrain2(s, mvb"<< O_C(0)<<", mvf"<< O_C(0)<<", mvb1"<< O_C(0) <<", mvf1"<< O_C(0) <<", thSAD = 6400, thSCD1 = 1800, thSCD2 = 980)" << std::endl;
        out << "degcuda = srcuda.KMDegrain2(scuda, mvb, mvf, mvb1, mvf1, thSAD = 6400, thSCD1 = 1800, thSCD2 = 980)"<< O_C(0) << std::endl;
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
        out << "degcuda = srcuda.KMDegrain1(scuda, mvb.OnCPU(0), mvf.OnCPU(0), thSAD = 6400, thSCD1 = 1800, thSCD2 = 980)" << O_C(0) << std::endl;
      }
      else if (N == 2) {
        out << "degref = src.KMDegrain2(s, mvb, mvf, mvb1, mvf1, thSAD = 6400, thSCD1 = 1800, thSCD2 = 980)" << std::endl;
        out << "degcuda = srcuda.KMDegrain2(scuda, mvb.OnCPU(0), mvf.OnCPU(0), mvb1.OnCPU(0), mvf1.OnCPU(0), thSAD = 6400, thSCD1 = 1800, thSCD2 = 980)" << O_C(0) << std::endl;
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

TEST_F(KTGMCTest, Degrain_1Blk8Pel2)
{
	DegrainTest(TF_MID, 1, 8, 2);
}

TEST_F(KTGMCTest, Degrain_1Blk8Pel1)
{
	DegrainTest(TF_MID, 1, 8, 1);
}

TEST_F(KTGMCTest, Degrain_1Blk16Pel2)
{
  DegrainTest(TF_MID, 1, 16, 2);
}

TEST_F(KTGMCTest, Degrain_1Blk16Pel1)
{
  DegrainTest(TF_MID, 1, 16, 1);
}

TEST_F(KTGMCTest, Degrain_1Blk32Pel2)
{
  DegrainTest(TF_MID, 1, 32, 2);
}

TEST_F(KTGMCTest, Degrain_1Blk32Pel1)
{
  DegrainTest(TF_MID, 1, 32, 1);
}

TEST_F(KTGMCTest, Degrain_2Blk8Pel2)
{
	DegrainTest(TF_MID, 2, 8, 2);
}

TEST_F(KTGMCTest, Degrain_2Blk8Pel1)
{
	DegrainTest(TF_MID, 2, 8, 1);
}

TEST_F(KTGMCTest, Degrain_2Blk16Pel2)
{
  DegrainTest(TF_MID, 2, 16, 2);
}

TEST_F(KTGMCTest, Degrain_2Blk16Pel1)
{
  DegrainTest(TF_MID, 2, 16, 1);
}

TEST_F(KTGMCTest, Degrain_2Blk32Pel2)
{
  DegrainTest(TF_MID, 2, 32, 2);
}

TEST_F(KTGMCTest, Degrain_2Blk32Pel1)
{
  DegrainTest(TF_MID, 2, 32, 1);
}

TEST_F(KTGMCTest, Degrain_2Blk16Pel2Begin)
{
  DegrainTest(TF_BEGIN, 2, 16, 2);
}

TEST_F(KTGMCTest, Degrain_2Blk16Pel1End)
{
  DegrainTest(TF_END, 2, 16, 2);
}

void KTGMCTest::DegrainBinomialTest(TEST_FRAMES tf, int N, int blksize, int pel)
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

    out << "bindeg = srcuda.KMDegrain2(scuda, mvb, mvf, mvb1, mvf1, thSAD = 6400, thSCD1 = 1800, thSCD2 = 980, binomial = true)" O_C(0) << std::endl;
    out << "deg1 = srcuda.KMDegrain1(scuda, mvb, mvf, thSAD = 6400, thSCD1 = 1800, thSCD2 = 980)" O_C(0) << std::endl;
    out << "deg2 = srcuda.KMDegrain1(scuda, mvb1, mvf1, thSAD = 6400, thSCD1 = 1800, thSCD2 = 980)" O_C(0) << std::endl;
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

TEST_F(KTGMCTest, DegrainBinomial_2Blk32Pel1)
{
  DegrainBinomialTest(TF_MID, 2, 32, 1);
}

#pragma endregion

#pragma region Compensate

void KTGMCTest::CompensateTest(TEST_FRAMES tf, int blksize, int pel)
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
      out << "comref = src.KMCompensate(s, mvb" O_C(0) ",thSCD1=1800,thSCD2=980)" << std::endl;
      out << "comcuda = srcuda.KMCompensate(scuda,mvb,thSCD1=1800,thSCD2=980)" O_C(0) "" << std::endl;
    }
    else {
      out << "mvb = s.KMAnalyse(isb = true, delta = 1, chroma = false, blksize = " << blksize <<
        ", overlap = " << (blksize / 2) << ", lambda = 400, global = true, meander = false, partial = pmvb)" << std::endl;
      out << "comref = src.KMCompensate(s, mvb, thSCD1=180,thSCD2=98)" << std::endl;
      out << "comcuda = srcuda.KMCompensate(scuda, mvb.OnCPU(0), thSCD1=180,thSCD2=98)" O_C(0) "" << std::endl;
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

TEST_F(KTGMCTest, Compensate_Blk8Pel2)
{
	CompensateTest(TF_MID, 8, 2);
}

TEST_F(KTGMCTest, Compensate_Blk8Pel1)
{
	CompensateTest(TF_MID, 8, 1);
}

TEST_F(KTGMCTest, Compensate_Blk16Pel2)
{
  CompensateTest(TF_MID, 16, 2);
}

TEST_F(KTGMCTest, Compensate_Blk16Pel1)
{
  CompensateTest(TF_MID, 16, 1);
}

TEST_F(KTGMCTest, Compensate_Blk32Pel2)
{
  CompensateTest(TF_MID, 32, 2);
}

TEST_F(KTGMCTest, Compensate_Blk32Pel1)
{
  CompensateTest(TF_MID, 32, 1);
}

TEST_F(KTGMCTest, Compensate_Blk16Pel2Begin)
{
  CompensateTest(TF_BEGIN, 16, 2);
}

TEST_F(KTGMCTest, Compensate_Blk16Pel2End)
{
  CompensateTest(TF_END, 16, 2);
}

#pragma endregion

#pragma region MVReplace

void KTGMCTest::MVReplaceTest(TEST_FRAMES tf, bool kvm)
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

TEST_F(KTGMCTest, MVReplace_KMV)
{
  MVReplaceTest(TF_MID, true);
}

TEST_F(KTGMCTest, MVReplace_MV)
{
  MVReplaceTest(TF_MID, false);
}

#pragma endregion

#pragma region Bob

void KTGMCTest::BobTest(TEST_FRAMES tf, bool parity)
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
    out << "cuda = srcuda.KTGMC_Bob( 0,0.5 )" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, BobTest_TFF)
{
  BobTest(TF_MID, true);
}

TEST_F(KTGMCTest, BobTest_BFF)
{
  BobTest(TF_MID, false);
}

#pragma endregion

#pragma region BinomialSoften

void KTGMCTest::BinomialSoftenTest(TEST_FRAMES tf, int radius, bool chroma)
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
    out << "cuda = srcuda.KBinomialTemporalSoften(" << radius << ", 28, " << (chroma ? "true" : "false") << ")" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, BinomialSoften_Rad1WithC)
{
  BinomialSoftenTest(TF_MID, 1, true);
}

TEST_F(KTGMCTest, BinomialSoften_Rad2WithC)
{
  BinomialSoftenTest(TF_MID, 2, true);
}

TEST_F(KTGMCTest, BinomialSoften_Rad1NoC)
{
  BinomialSoftenTest(TF_MID, 1, false);
}

TEST_F(KTGMCTest, BinomialSoften_Rad2NoC)
{
  BinomialSoftenTest(TF_MID, 2, false);
}

#pragma endregion

#pragma region RemoveGrain

void KTGMCTest::RemoveGrainTest(TEST_FRAMES tf, int mode, bool chroma)
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
    out << "cuda = srcuda.KRemoveGrain(" << mode << (chroma ? "" : ", -1") << ")" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, RemoveGrain_Mode1WithC)
{
  RemoveGrainTest(TF_MID, 1, true);
}

TEST_F(KTGMCTest, RemoveGrain_Mode1NoC)
{
  RemoveGrainTest(TF_MID, 1, false);
}

TEST_F(KTGMCTest, RemoveGrain_Mode2WithC)
{
  RemoveGrainTest(TF_MID, 2, true);
}

TEST_F(KTGMCTest, RemoveGrain_Mode2NoC)
{
  RemoveGrainTest(TF_MID, 2, false);
}

TEST_F(KTGMCTest, RemoveGrain_Mode3WithC)
{
  RemoveGrainTest(TF_MID, 3, true);
}

TEST_F(KTGMCTest, RemoveGrain_Mode3NoC)
{
  RemoveGrainTest(TF_MID, 3, false);
}

TEST_F(KTGMCTest, RemoveGrain_Mode4WithC)
{
  RemoveGrainTest(TF_MID, 4, true);
}

TEST_F(KTGMCTest, RemoveGrain_Mode4NoC)
{
  RemoveGrainTest(TF_MID, 4, false);
}

TEST_F(KTGMCTest, RemoveGrain_Mode12WithC)
{
  RemoveGrainTest(TF_MID, 12, true);
}

TEST_F(KTGMCTest, RemoveGrain_Mode12NoC)
{
  RemoveGrainTest(TF_MID, 12, false);
}

TEST_F(KTGMCTest, RemoveGrain_Mode20WithC)
{
  RemoveGrainTest(TF_MID, 20, true);
}

TEST_F(KTGMCTest, RemoveGrain_Mode20NoC)
{
  RemoveGrainTest(TF_MID, 20, false);
}

#pragma endregion

#pragma region Repair

void KTGMCTest::RepairTest(TEST_FRAMES tf, int mode, bool chroma)
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
    out << "cuda = srcuda.KRepair(srefcuda, " << mode << (chroma ? "" : ", -1") << ")" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, Repair_Mode1WithC)
{
  RepairTest(TF_MID, 1, true);
}

TEST_F(KTGMCTest, Repair_Mode1NoC)
{
  RepairTest(TF_MID, 1, false);
}

TEST_F(KTGMCTest, Repair_Mode2WithC)
{
  RepairTest(TF_MID, 2, true);
}

TEST_F(KTGMCTest, Repair_Mode2NoC)
{
  RepairTest(TF_MID, 2, false);
}

TEST_F(KTGMCTest, Repair_Mode3WithC)
{
  RepairTest(TF_MID, 3, true);
}

TEST_F(KTGMCTest, Repair_Mode3NoC)
{
  RepairTest(TF_MID, 3, false);
}

TEST_F(KTGMCTest, Repair_Mode4WithC)
{
  RepairTest(TF_MID, 4, true);
}

TEST_F(KTGMCTest, Repair_Mode4NoC)
{
  RepairTest(TF_MID, 4, false);
}

#pragma endregion

#pragma region VerticalCleaner

void KTGMCTest::VerticalCleanerTest(TEST_FRAMES tf, int mode, bool chroma)
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
    out << "cuda = srcuda.KVerticalCleaner(" << mode << (chroma ? "" : ", 0") << ")" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, VerticalCleaner_WithC)
{
  VerticalCleanerTest(TF_MID, 1, true);
}

TEST_F(KTGMCTest, VerticalCleaner_NoC)
{
  VerticalCleanerTest(TF_MID, 1, false);
}

#pragma endregion

#pragma region GaussResize

void KTGMCTest::GaussResizeTest(TEST_FRAMES tf, bool chroma)
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
    out << "cuda = srcuda.KGaussResize(p=2" << (chroma ? "" : ", chroma=false") << ")" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, GaussResizeTest_WithC)
{
  GaussResizeTest(TF_MID, true);
}

TEST_F(KTGMCTest, GaussResizeTest_NoC)
{
  GaussResizeTest(TF_MID, false);
}

#pragma endregion

#pragma region InpandVerticalX2

void KTGMCTest::InpandVerticalX2Test(TEST_FRAMES tf, bool chroma)
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
    out << "cuda = srcuda.KInpandVerticalX2(U = " << rc << ", V = " << rc << ")" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, InpandVerticalX2Test_WithC)
{
  InpandVerticalX2Test(TF_MID, true);
}

TEST_F(KTGMCTest, InpandVerticalX2Test_NoC)
{
  InpandVerticalX2Test(TF_MID, false);
}

#pragma endregion

#pragma region ExpandVerticalX2

void KTGMCTest::ExpandVerticalX2Test(TEST_FRAMES tf, bool chroma)
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
    out << "cuda = srcuda.KExpandVerticalX2(U = " << rc << ", V = " << rc << ")" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, ExpandVerticalX2Test_WithC)
{
  ExpandVerticalX2Test(TF_MID, true);
}

TEST_F(KTGMCTest, ExpandVerticalX2Test_NoC)
{
  ExpandVerticalX2Test(TF_MID, false);
}

#pragma endregion

#pragma region MakeDiff

void KTGMCTest::MakeDiffTest(TEST_FRAMES tf, bool chroma, bool makediff)
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
      out << "cuda = srcuda.KMakeDiff(sr2cuda, U = " << rc << ", V = " << rc << ")" O_C(0) "" << std::endl;
    }
    else {
      out << "ref = src.mt_adddiff(src2, U=" << rc << ",V=" << rc << ")" << std::endl;
      out << "cuda = srcuda.KAddDiff(sr2cuda, U = " << rc << ", V = " << rc << ")" O_C(0) "" << std::endl;
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

TEST_F(KTGMCTest, MakeDiffTest_WithC)
{
  MakeDiffTest(TF_MID, true, true);
}

TEST_F(KTGMCTest, MakeDiffTest_NoC)
{
  MakeDiffTest(TF_MID, false, true);
}

TEST_F(KTGMCTest, AddDiffTest_WithC)
{
  MakeDiffTest(TF_MID, true, false);
}

TEST_F(KTGMCTest, AddDiffTest_NoC)
{
  MakeDiffTest(TF_MID, false, false);
}

#pragma endregion

#pragma region Logic

void KTGMCTest::LogicTest(TEST_FRAMES tf, const char* mode, bool chroma)
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
    out << "cuda = srcuda.KLogic(sr2cuda, mode=\"" << mode << "\", U = " << rc << ", V = " << rc << ")" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, LogicTest_MinWithC)
{
  LogicTest(TF_MID, "min", true);
}

TEST_F(KTGMCTest, LogicTest_MinNoC)
{
  LogicTest(TF_MID, "min", false);
}

TEST_F(KTGMCTest, LogicTest_MaxWithC)
{
  LogicTest(TF_MID, "max", true);
}

TEST_F(KTGMCTest, LogicTest_MaxNoC)
{
  LogicTest(TF_MID, "max", false);
}

#pragma endregion

#pragma region BobShimmerFixesMerge

void KTGMCTest::BobShimmerFixesMergeTest(TEST_FRAMES tf, int rep, bool chroma)
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
    out << "cuda = srcuda.KTGMC_KeepOnlyBobShimmerFixes(sr2cuda, " << rep << ", " << chromastr << ")" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, BobShimmerFixesMergeTest_Rep3WithC)
{
  BobShimmerFixesMergeTest(TF_MID, 3, true);
}

TEST_F(KTGMCTest, BobShimmerFixesMergeTest_Rep3NoC)
{
  BobShimmerFixesMergeTest(TF_MID, 3, false);
}

TEST_F(KTGMCTest, BobShimmerFixesMergeTest_Rep4WithC)
{
  BobShimmerFixesMergeTest(TF_MID, 4, true);
}

TEST_F(KTGMCTest, BobShimmerFixesMergeTest_Rep4NoC)
{
  BobShimmerFixesMergeTest(TF_MID, 4, false);
}

#pragma endregion

#pragma region VResharpen

void KTGMCTest::VResharpenTest(TEST_FRAMES tf)
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
    out << "cuda = srcuda.KTGMC_VResharpen(U=3, V=3)" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, VResharpenTest_WithC)
{
  VResharpenTest(TF_MID);
}

#pragma endregion

#pragma region Resharpen

void KTGMCTest::ResharpenTest(TEST_FRAMES tf)
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
    out << "cuda = srcuda.KTGMC_Resharpen(sr1cuda, 0.70000, U=3, V=3)" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, ResharpenTest_WithC)
{
  ResharpenTest(TF_MID);
}

#pragma endregion

#pragma region LimitOverSharpen

void KTGMCTest::LimitOverSharpenTest(TEST_FRAMES tf)
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

    out << "cuda = srcuda.KTGMC_LimitOverSharpen(sr1cuda, sr2cuda, sr3cuda, 0, U=3, V=3)" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, LimitOverSharpenTest_WithC)
{
  LimitOverSharpenTest(TF_MID);
}

#pragma endregion

#pragma region ToFullRange

void KTGMCTest::ToFullRangeTest(TEST_FRAMES tf, bool chroma)
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

    out << "cuda = srcuda.KTGMC_ToFullRange(u=" << rc << ",v=" << rc << ")" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, ToFullRangeTest_WithC)
{
  ToFullRangeTest(TF_MID, true);
}

TEST_F(KTGMCTest, ToFullRangeTest_NoC)
{
  ToFullRangeTest(TF_MID, false);
}

#pragma endregion

#pragma region TweakSearchClip

void KTGMCTest::TweakSearchClipTest(TEST_FRAMES tf, bool chroma)
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

    out << "cuda = repair0cuda.KTGMC_TweakSearchClip(bobbedcuda, spatialBlurcuda, u=" << rc << ",v=" << rc << ")" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, TweakSearchClipTest_WithC)
{
  TweakSearchClipTest(TF_MID, true);
}

TEST_F(KTGMCTest, TweakSearchClipTest_NoC)
{
  TweakSearchClipTest(TF_MID, false);
}

#pragma endregion

#pragma region LosslessProc

void KTGMCTest::LosslessProcTest(TEST_FRAMES tf, bool chroma)
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
    out << "cuda = KTGMC_LosslessProc(xcuda, ycuda, u=" << rc << ",v=" << rc << ")" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, LosslessProcTest_WithC)
{
  LosslessProcTest(TF_MID, true);
}

TEST_F(KTGMCTest, LosslessProcTest_NoC)
{
  LosslessProcTest(TF_MID, false);
}

#pragma endregion

#pragma region Merge

void KTGMCTest::MergeTest(TEST_FRAMES tf, bool chroma)
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
      out << "cuda = srcuda.KMerge(sr1cuda, 0.844)" O_C(0) "" << std::endl;
    }
    else {
      out << "ref = src.MergeLuma(src1, 0.844)" << std::endl;
      out << "cuda = srcuda.KMergeLuma(sr1cuda, 0.844)" O_C(0) "" << std::endl;
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

TEST_F(KTGMCTest, MergeTest_WithC)
{
  MergeTest(TF_MID, true);
}

TEST_F(KTGMCTest, MergeTest_NoC)
{
  MergeTest(TF_MID, false);
}

#pragma endregion

#pragma region Weave

void KTGMCTest::WeaveTest(TEST_FRAMES tf, bool parity, bool dbl)
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
      out << "cuda = srcuda.SeparateFields().SelectEvery( 4, 0,3 ).KDoubleWeave()" O_C(0) "" << std::endl;
    }
    else {
      out << "ref = src.SeparateFields().SelectEvery( 4, 0,3 ).Weave()" << std::endl;
      out << "cuda = srcuda.SeparateFields().SelectEvery( 4, 0,3 ).KWeave()" O_C(0) "" << std::endl;
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

TEST_F(KTGMCTest, WeaveTest_DoubleTFF)
{
  WeaveTest(TF_MID, true, true);
}

TEST_F(KTGMCTest, WeaveTest_DoubleBFF)
{
  WeaveTest(TF_MID, false, true);
}

TEST_F(KTGMCTest, WeaveTest_TFF)
{
  WeaveTest(TF_MID, true, false);
}

TEST_F(KTGMCTest, WeaveTest_BFF)
{
  WeaveTest(TF_MID, false, false);
}

#pragma endregion

#pragma region Copy

void KTGMCTest::CopyTest(TEST_FRAMES tf, bool cuda)
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
      out << "cuda = srcuda.SeparateFields().SelectEvery( 4, 0,3 ).KCopy()" O_C(0) "" << std::endl;
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

TEST_F(KTGMCTest, CopyTest_CUDA)
{
  CopyTest(TF_MID, true);
}

TEST_F(KTGMCTest, CopyTest_CPU)
{
  CopyTest(TF_MID, false);
}

#pragma endregion

#pragma region ErrorAdjust

void KTGMCTest::ErrorAdjustTest(TEST_FRAMES tf, bool chroma)
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
    out << "cuda = srcuda.KTGMC_ErrorAdjust(sr2cuda, " << errorAdj << ", U=" << rc << ", V = " << rc << ")" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, ErrorAdjustTest_WithC)
{
  ErrorAdjustTest(TF_MID, true);
}

#pragma endregion

#pragma region NNEDI3

void KTGMCTest::NNEDI3Test(TEST_FRAMES tf, bool chroma, int nsize, int nns, int qual, int pscrn)
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
      "pscrn=" << pscrn << ",U=" << UV << ",V=" << UV << ")" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, NNEDI3Test_NS0NN0Q1PS2)
{
  NNEDI3Test(TF_MID, true, 0, 0, 1, 2);
}

TEST_F(KTGMCTest, NNEDI3Test_NS1NN0Q1PS2)
{
  NNEDI3Test(TF_MID, true, 1, 0, 1, 2);
}

TEST_F(KTGMCTest, NNEDI3Test_NS2NN0Q1PS2)
{
  NNEDI3Test(TF_MID, true, 2, 0, 1, 2);
}

TEST_F(KTGMCTest, NNEDI3Test_NS3NN0Q1PS2)
{
  NNEDI3Test(TF_MID, true, 3, 0, 1, 2);
}

TEST_F(KTGMCTest, NNEDI3Test_NS4NN0Q1PS2)
{
  NNEDI3Test(TF_MID, true, 4, 0, 1, 2);
}

TEST_F(KTGMCTest, NNEDI3Test_NS5NN0Q1PS2)
{
  NNEDI3Test(TF_MID, true, 5, 0, 1, 2);
}

TEST_F(KTGMCTest, NNEDI3Test_NS6NN0Q1PS2)
{
  NNEDI3Test(TF_MID, true, 6, 0, 1, 2);
}

TEST_F(KTGMCTest, NNEDI3Test_NS0NN1Q1PS2)
{
  NNEDI3Test(TF_MID, true, 0, 1, 1, 2);
}

TEST_F(KTGMCTest, NNEDI3Test_NS0NN2Q1PS2)
{
  NNEDI3Test(TF_MID, true, 0, 2, 1, 2);
}

TEST_F(KTGMCTest, NNEDI3Test_NS0NN3Q1PS2)
{
  NNEDI3Test(TF_MID, true, 0, 3, 1, 2);
}

TEST_F(KTGMCTest, NNEDI3Test_NS0NN4Q1PS2)
{
  NNEDI3Test(TF_MID, true, 0, 4, 1, 2);
}

TEST_F(KTGMCTest, NNEDI3Test_NS0NN0Q2PS2)
{
  NNEDI3Test(TF_MID, true, 0, 0, 2, 2);
}

// 性能評価用
TEST_F(KTGMCTest, NNEDI3Test_NS1NN1Q1PS2)
{
  NNEDI3Test(TF_MID, true, 1, 1, 1, 2);
}

TEST_F(KTGMCTest, NNEDI3Test_NoC)
{
  NNEDI3Test(TF_MID, false, 0, 0, 1, 2);
}

// 性能評価用
TEST_F(KTGMCTest, NNEDI3Test_Perf)
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
    out << "return srcuda.KNNEDI3(field=-2,nsize=1,nns=1,qual=1,pscrn=2,opt=1,threads=1,U=" << UV << ",V=" << UV << ")" O_C(2) << std::endl;
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

TEST_F(KTGMCTest, DeviceCheck)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);
    out << "LWLibavVideoSource(\"test.ts\")" O_C(0) "" << std::endl;
    out.close();

    EXPECT_THROW(env->Invoke("Import", scriptpath.c_str()).AsClip(), AvisynthError);
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

TEST_F(KTGMCTest, StartupTime)
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
TEST_F(KTGMCTest, KTGMC_Perf)
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
    out << "src.OnCPU(2).KTGMC()" O_C(2) << std::endl;

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

TEST_F(KTGMCTest, MemoryLeak)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string scriptpath = workDirPath + "\\script.avs";
    std::ofstream out(scriptpath);

    out << "SetLogParams(\"avsrun.log\", LOG_WARNING)" << std::endl;
    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "return src.OnCPU(1)" O_C(0) "" << std::endl;
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

TEST_F(KTGMCTest, DeviceMatchingBug)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string scriptpath = workDirPath + "\\script.avs";
    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "src.OnCPU(0)" O_C(0) ".AudioDub(LWLibavAudioSource(\"test.ts\"))" << std::endl;
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

TEST_F(KTGMCTest, AnalyzeStaticTest)
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
    out << "cuda = srcuda.KAnalyzeStatic(30, 15)" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, MergeStaticTest)
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
    out << "cuda = bbcuda.KMergeStatic(srcuda, stt.OnCPU(0))" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, AnalyzeFrameTest)
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
    out << "cuda = srcuda.KFMFrameAnalyze()" O_C(0) "" << std::endl;

    out << "KFMFrameAnalyzeCheck(ref, cuda)" << std::endl;

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

TEST_F(KTGMCTest, TelecineTest)
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
    out << "cuda = srcuda.KTelecine(fm.OnCPU(0))" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, RemoveCombeTest)
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

		out << "ref = src.KRemoveCombe(6, 50)" << std::endl;
		out << "cuda = srcuda.KRemoveCombe(6, 50)" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, SwitchTest)
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
		out << "fmcuda = fm.OnCPU(0)" << std::endl;
		out << "tc = src.KTelecine(fm)" << std::endl;
		out << "tc = tc.KRemoveCombe(6, 50)" << std::endl;
		out << "tccuda = tc.OnCPU(0)" << std::endl;
		out << "bb = src.Bob()" << std::endl;
		out << "bbcuda = bb.OnCPU(0)" << std::endl;

		out << "ref = bb.KFMSwitch(tc, fm, tc, 10000, 40)" << std::endl;
		out << "cuda = bbcuda.KFMSwitch(tccuda, fmcuda, tccuda, 10000, 40)" O_C(0) "" << std::endl;

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

#pragma region TemporalNR

void KTGMCTest::TemporalNRTest(TEST_FRAMES tf)
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

		out << "ref = src.KTemporalNR(3, 4)" << std::endl;
		out << "cuda = srcuda.KTemporalNR(3, 4)" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, TemporalNRTest)
{
	TemporalNRTest(TF_MID);
}

#pragma endregion

#pragma region Deband

void KTGMCTest::DebandTest(TEST_FRAMES tf, int sample_mode, bool blur_first)
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

		const char* blur_str = blur_first ? "true" : "false";

		out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
		out << "srcuda = src.OnCPU(0)" << std::endl;

		out << "ref = src.KDeband(25, 4, " << sample_mode << ", " << blur_str << ")" << std::endl;
		out << "cuda = srcuda.KDeband(25, 4, " << sample_mode << ", " << blur_str << ")" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, DebandTest_Mode0F)
{
	DebandTest(TF_MID, 0, false);
}

TEST_F(KTGMCTest, DebandTest_Mode1F)
{
	DebandTest(TF_MID, 1, false);
}

TEST_F(KTGMCTest, DebandTest_Mode2F)
{
	DebandTest(TF_MID, 2, false);
}

TEST_F(KTGMCTest, DebandTest_Mode0T)
{
	DebandTest(TF_MID, 0, true);
}

TEST_F(KTGMCTest, DebandTest_Mode1T)
{
	DebandTest(TF_MID, 1, true);
}

TEST_F(KTGMCTest, DebandTest_Mode2T)
{
	DebandTest(TF_MID, 2, true);
}

#pragma endregion

#pragma region EdgeLevel

void KTGMCTest::EdgeLevelTest(TEST_FRAMES tf, int repair, bool chroma)
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

		out << "ref = src.KEdgeLevel(16, 10, " << repair << ", uv=" << (chroma ? "true" : "false") << ")" << std::endl;
		out << "cuda = srcuda.KEdgeLevel(16, 10, " << repair << ", uv=" << (chroma ? "true" : "false") << ")" O_C(0) "" << std::endl;

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

TEST_F(KTGMCTest, EdgeLevel_Rep0WithC)
{
	EdgeLevelTest(TF_MID, 0, true);
}

TEST_F(KTGMCTest, EdgeLevel_Rep1WithC)
{
	EdgeLevelTest(TF_MID, 1, true);
}

TEST_F(KTGMCTest, EdgeLevel_Rep2WithC)
{
	EdgeLevelTest(TF_MID, 2, true);
}

TEST_F(KTGMCTest, EdgeLevel_Rep0NoC)
{
	EdgeLevelTest(TF_MID, 0, false);
}

TEST_F(KTGMCTest, EdgeLevel_Rep1NoC)
{
	EdgeLevelTest(TF_MID, 1, false);
}

TEST_F(KTGMCTest, EdgeLevel_Rep2NoC)
{
	EdgeLevelTest(TF_MID, 2, false);
}

#pragma endregion

#pragma region CFieldDiff

void KTGMCTest::CFieldDiffTest(int nt, bool chroma)
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

      out << "current_frame = 100" << std::endl;
      out << "ref = src.CFieldDiff(nt = " << nt << ", chroma=" << (chroma ? "true" : "false") << ")" << std::endl;
      out << "cuda = EvalOnCUDA(\"srcuda.KCFieldDiff(nt = " << nt << ", chroma=" << (chroma ? "true" : "false") << ")\")" << std::endl;

      out.close();

      {
         env->Invoke("Import", scriptpath.c_str());
         double ref = env->GetVar("ref").AsFloat();
         double cuda = env->GetVar("cuda").AsFloat();
         // 境界の扱いが異なるので（多分）一致しない
         // 差が1%未満であることを確認
         if (std::abs(ref - cuda) / ref >= 0.01) {
            printf("誤差が大きすぎます %f vs %f\n", ref, cuda);
            GTEST_FAIL();
         }
      }
   }
   catch (const AvisynthError& err) {
      printf("%s\n", err.msg);
      GTEST_FAIL();
   }
}

TEST_F(KTGMCTest, CFieldDiff_Nt0WithC)
{
   CFieldDiffTest(0, true);
}

TEST_F(KTGMCTest, CFieldDiff_Nt0NoC)
{
   CFieldDiffTest(0, false);
}

TEST_F(KTGMCTest, CFieldDiff_Nt3WithC)
{
   CFieldDiffTest(3, true);
}

TEST_F(KTGMCTest, CFieldDiff_Nt3NoC)
{
   CFieldDiffTest(3, false);
}

#pragma endregion

#pragma region CFrameDiffDup

void KTGMCTest::CFrameDiffDupTest(int blocksize, bool chroma)
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

    out << "current_frame = 100" << std::endl;
    out << "ref = src.KCFrameDiffDup(blksize = " << blocksize << ", chroma=" << (chroma ? "true" : "false") << ")" << std::endl;
    out << "cuda = EvalOnCUDA(\"srcuda.KCFrameDiffDup(blksize = " << blocksize << ", chroma=" << (chroma ? "true" : "false") << ")\")" << std::endl;

    out.close();

    {
      env->Invoke("Import", scriptpath.c_str());
      double ref = env->GetVar("ref").AsFloat();
      double cuda = env->GetVar("cuda").AsFloat();
      // 境界の扱いが異なるので（多分）一致しない
      // 差が1%未満であることを確認
      if (std::abs(ref - cuda) / ref >= 0.01) {
        printf("誤差が大きすぎます %f vs %f\n", ref, cuda);
        GTEST_FAIL();
      }
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

TEST_F(KTGMCTest, CFrameDiffDup_Blk32WithC)
{
  CFrameDiffDupTest(32, true);
}

TEST_F(KTGMCTest, CFrameDiffDup_Blk32NoC)
{
  CFrameDiffDupTest(32, false);
}

TEST_F(KTGMCTest, CFrameDiffDup_Blk8WithC)
{
  CFrameDiffDupTest(8, true);
}

TEST_F(KTGMCTest, CFrameDiffDup_Blk8NoC)
{
  CFrameDiffDupTest(8, false);
}

#pragma endregion


TEST_F(KTGMCTest, AvsProp)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "LWLibavVideoSource(\"test.ts\")" << std::endl;

    out << "AddProp(\"luma\", \"AverageLuma()\")" << std::endl;
    out << "ScriptClip(\"\"\"subtitle(string(getprop(\"luma\")))\"\"\")" << std::endl;

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

TEST_F(KTGMCTest, DISABLED_DumpAVSProperty)
{
	PEnv env;
	try {
		env = PEnv(CreateScriptEnvironment2());

		int types[] = {
			VideoInfo::CS_BGR24,
			VideoInfo::CS_BGR32,
			VideoInfo::CS_YUY2,
			VideoInfo::CS_YV24,
			VideoInfo::CS_YV16,
			VideoInfo::CS_YV12,
			VideoInfo::CS_I420,
			VideoInfo::CS_IYUV,
			VideoInfo::CS_YV411,
			VideoInfo::CS_Y8,
			VideoInfo::CS_YUV444P10,
			VideoInfo::CS_YUV422P10,
			VideoInfo::CS_YUV420P10,
			VideoInfo::CS_Y10,
			VideoInfo::CS_YUV444P12,
			VideoInfo::CS_YUV422P12,
			VideoInfo::CS_YUV420P12,
			VideoInfo::CS_Y12,
			VideoInfo::CS_YUV444P14,
			VideoInfo::CS_YUV422P14,
			VideoInfo::CS_YUV420P14,
			VideoInfo::CS_Y14,
			VideoInfo::CS_YUV444P16,
			VideoInfo::CS_YUV422P16,
			VideoInfo::CS_YUV420P16,
			VideoInfo::CS_Y16,
			VideoInfo::CS_YUV444PS,
			VideoInfo::CS_YUV422PS,
			VideoInfo::CS_YUV420PS,
			VideoInfo::CS_Y32,
			VideoInfo::CS_BGR48,
			VideoInfo::CS_BGR64,
			VideoInfo::CS_RGBP,
			VideoInfo::CS_RGBP10,
			VideoInfo::CS_RGBP12,
			VideoInfo::CS_RGBP14,
			VideoInfo::CS_RGBP16,
			VideoInfo::CS_RGBPS,
			VideoInfo::CS_RGBAP,
			VideoInfo::CS_RGBAP10,
			VideoInfo::CS_RGBAP12,
			VideoInfo::CS_RGBAP14,
			VideoInfo::CS_RGBAP16,
			VideoInfo::CS_RGBAPS,
			VideoInfo::CS_YUVA444,
			VideoInfo::CS_YUVA422,
			VideoInfo::CS_YUVA420,
			VideoInfo::CS_YUVA444P10,
			VideoInfo::CS_YUVA422P10,
			VideoInfo::CS_YUVA420P10,
			VideoInfo::CS_YUVA444P12,
			VideoInfo::CS_YUVA422P12,
			VideoInfo::CS_YUVA420P12,
			VideoInfo::CS_YUVA444P14,
			VideoInfo::CS_YUVA422P14,
			VideoInfo::CS_YUVA420P14,
			VideoInfo::CS_YUVA444P16,
			VideoInfo::CS_YUVA422P16,
			VideoInfo::CS_YUVA420P16,
			VideoInfo::CS_YUVA444PS,
			VideoInfo::CS_YUVA422PS,
			VideoInfo::CS_YUVA420PS
		};
		const char* typenames[] = {
			"CS_BGR24",
			"CS_BGR32",
			"CS_YUY2",
			"CS_YV24",
			"CS_YV16",
			"CS_YV12",
			"CS_I420",
			"CS_IYUV",
			"CS_YV411",
			"CS_Y8",
			"CS_YUV444P10",
			"CS_YUV422P10",
			"CS_YUV420P10",
			"CS_Y10",
			"CS_YUV444P12",
			"CS_YUV422P12",
			"CS_YUV420P12",
			"CS_Y12",
			"CS_YUV444P14",
			"CS_YUV422P14",
			"CS_YUV420P14",
			"CS_Y14",
			"CS_YUV444P16",
			"CS_YUV422P16",
			"CS_YUV420P16",
			"CS_Y16",
			"CS_YUV444PS",
			"CS_YUV422PS",
			"CS_YUV420PS",
			"CS_Y32",
			"CS_BGR48",
			"CS_BGR64",
			"CS_RGBP",
			"CS_RGBP10",
			"CS_RGBP12",
			"CS_RGBP14",
			"CS_RGBP16",
			"CS_RGBPS",
			"CS_RGBAP",
			"CS_RGBAP10",
			"CS_RGBAP12",
			"CS_RGBAP14",
			"CS_RGBAP16",
			"CS_RGBAPS",
			"CS_YUVA444",
			"CS_YUVA422",
			"CS_YUVA420",
			"CS_YUVA444P10",
			"CS_YUVA422P10",
			"CS_YUVA420P10",
			"CS_YUVA444P12",
			"CS_YUVA422P12",
			"CS_YUVA420P12",
			"CS_YUVA444P14",
			"CS_YUVA422P14",
			"CS_YUVA420P14",
			"CS_YUVA444P16",
			"CS_YUVA422P16",
			"CS_YUVA420P16",
			"CS_YUVA444PS",
			"CS_YUVA422PS",
			"CS_YUVA420PS"
		};

		for (int i = 0; i < sizeof(types) / sizeof(types[0]); ++i) {
			VideoInfo vi = VideoInfo();
			vi.width = 100;
			vi.height = 32;
			vi.pixel_type = types[i];

			printf("Pixel Format: %s\n", typenames[i]);
			printf("VideoInfo Properties\n");
			printf("IsRGB() = %d\n", vi.IsRGB());
			printf("IsRGB24() = %d\n", vi.IsRGB24());
			printf("IsRGB32() = %d\n", vi.IsRGB32());
			printf("IsYUV() = %d\n", vi.IsYUV());
			printf("IsYUY2() = %d\n", vi.IsYUY2());
			printf("IsYV24() = %d\n", vi.IsYV24());
			printf("IsYV16() = %d\n", vi.IsYV16());
			printf("IsYV12() = %d\n", vi.IsYV12());
			printf("IsYV411() = %d\n", vi.IsYV411());
			printf("IsY8() = %d\n", vi.IsY8());
			printf("IsPlanar() = %d\n", vi.IsPlanar());
			printf("BytesFromPixels(1) = %d\n", vi.BytesFromPixels(1));
			printf("RowSize() = %d\n", vi.RowSize());
			printf("BitsPerPixel() = %d\n", vi.BitsPerPixel());
			printf("NumComponents() = %d\n", vi.NumComponents());
			printf("ComponentSize() = %d\n", vi.ComponentSize());
			printf("BitsPerComponent() = %d\n", vi.BitsPerComponent());
			printf("Is444() = %d\n", vi.Is444());
			printf("Is422() = %d\n", vi.Is422());
			printf("Is420() = %d\n", vi.Is420());
			printf("IsY() = %d\n", vi.IsY());
			printf("IsRGB48() = %d\n", vi.IsRGB48());
			printf("IsRGB64() = %d\n", vi.IsRGB64());
			printf("IsYUVA() = %d\n", vi.IsYUVA());
			printf("IsPlanarRGB() = %d\n", vi.IsPlanarRGB());
			printf("IsPlanarRGBA() = %d\n", vi.IsPlanarRGBA());

			PVideoFrame frame = env->NewVideoFrame(vi);
			printf("VideoFrame Properties\n");
			printf("GetPitch() = %d\n", frame->GetPitch());
			printf("GetRowSize() = %d\n", frame->GetRowSize());
			printf("GetHeight() = %d\n", frame->GetHeight());
			printf("\n");
		}
	}
	catch (const AvisynthError& err) {
		printf("%s\n", err.msg);
		GTEST_FAIL();
	}
}
