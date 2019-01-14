
#include "TestCommons.h"

// テスト対象となるクラス Foo のためのフィクスチャ
class KFMTest : public AvsTestBase {
protected:
  KFMTest() { }

  void TemporalNRTest(TEST_FRAMES tf);
  void DebandTest(TEST_FRAMES tf, int sample_mode, bool blur_first);
  void EdgeLevelTest(TEST_FRAMES tf, int repair, bool chroma);

  void CFieldDiffTest(int nt, bool chroma);
  void CFrameDiffDupTest(int blocksize, bool chroma);

  void DecombUCF24Test(TEST_FRAMES tf, int chroma, bool show);

  void DeblockTest(TEST_FRAMES tf, int quality, bool is_soft);
};

#pragma region MergeStatic

TEST_F(KFMTest, AnalyzeStaticTest)
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

TEST_F(KFMTest, AnalyzeStaticSuperTest)
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

    out << "src = LWLibavVideoSource(\"test.ts\").OnCPU(0)" << std::endl;
    out << "super = src.KFMSuper()" << std::endl;

    out << "ref = src.KAnalyzeStatic(30, 15)" << std::endl;
    out << "cuda = src.KAnalyzeStatic(30, 15, super=super)" O_C(0) "" << std::endl;

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

TEST_F(KFMTest, MergeStaticTest)
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

TEST_F(KFMTest, PadTest)
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

    out << "src = LWLibavVideoSource(\"test.ts\").OnCPU(0)" << std::endl;

    out << "ref = src.KFMPad()" << std::endl;
    out << "cuda = src.KFMPad()" O_C(0) "" << std::endl;

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

TEST_F(KFMTest, KFMSuperTest)
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

    out << "src = LWLibavVideoSource(\"test.ts\").OnCPU(0)" << std::endl;
    out << "pad = src.KFMPad().OnCPU(0)" << std::endl;

    out << "ref = src.KFMSuper(pad)" << std::endl;
    out << "cuda = src.KFMSuper(pad)" O_C(0) "" << std::endl;

    // データのない部分除外
    out << "ImageCompare(ref, cuda, 1, offX=2, offY=1)" << std::endl;

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

TEST_F(KFMTest, PreCycleAnalyzeTest)
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

    out << "src = LWLibavVideoSource(\"test.ts\").OnCPU(0)" << std::endl;
    out << "super = src.KFMSuper(src.KFMPad()).OnCPU(0)" << std::endl;

    out << "ref = super.KPreCycleAnalyze(10, 5, 8, 4)" << std::endl;
    out << "cuda = super.KPreCycleAnalyze(10, 5, 8, 4)" O_C(0) "" << std::endl;

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

TEST_F(KFMTest, TelecineTest)
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

    out << "src = LWLibavVideoSource(\"test.ts\").OnCPU(0)" << std::endl;
    out << "fm = src.KFMSuper(src.KFMPad()).KPreCycleAnalyze().KFMCycleAnalyze(src).OnCPU(0)" << std::endl;

    out << "ref = src.KTelecine(fm)" << std::endl;
    out << "cuda = src.KTelecine(fm)" O_C(0) "" << std::endl;

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

TEST_F(KFMTest, TelecineSuperTest)
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

    out << "src = LWLibavVideoSource(\"test.ts\").OnCPU(0)" << std::endl;
    out << "super = src.KFMSuper(src.KFMPad()).OnCPU(0)" << std::endl;
    out << "fm = super.KPreCycleAnalyze().KFMCycleAnalyze(src).OnCPU(0)" << std::endl;

    out << "ref = super.KTelecineSuper(fm)" << std::endl;
    out << "cuda = super.KTelecineSuper(fm)" O_C(0) "" << std::endl;

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

TEST_F(KFMTest, SwitchFlagTest)
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

    out << "src = LWLibavVideoSource(\"test.ts\").OnCPU(0)" << std::endl;
    out << "super = src.KFMSuper(src.KFMPad()).OnCPU(0)" << std::endl;

    out << "ref = super.KSwitchFlag()" << std::endl;
    out << "cuda = super.KSwitchFlag()" O_C(0) "" << std::endl;

    // 余白は除く
    out << "ImageCompare(ref.Crop(4,2,-4,-2), cuda.Crop(4,2,-4,-2), 1)" << std::endl;

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

TEST_F(KFMTest, CombeMaskTest)
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

    out << "src = LWLibavVideoSource(\"test.ts\").OnCPU(0)" << std::endl;
    out << "flag = src.KFMSuper(src.KFMPad()).KSwitchFlag().OnCPU(0)" << std::endl;

    out << "ref = src.KCombeMask(flag)" << std::endl;
    out << "cuda = src.KCombeMask(flag)" O_C(0) "" << std::endl;

    // データはYとUのみ
    out << "ImageCompare(ref, cuda, 1, chroma=False)" << std::endl;
    out << "ImageCompare(ref.ExtractU(), cuda.ExtractU(), 1, chroma=False)" << std::endl;

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

TEST_F(KFMTest, RemoveCombeTest)
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

    out << "src = LWLibavVideoSource(\"test.ts\").OnCPU(0)" << std::endl;
    out << "super = src.KFMSuper(src.KFMPad())" << std::endl;

    out << "ref = src.KRemoveCombe(super)" << std::endl;
    out << "cuda = src.KRemoveCombe(super)" O_C(0) "" << std::endl;

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

TEST_F(KFMTest, PatchCombeTest)
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

    out << "src = LWLibavVideoSource(\"test.ts\").OnCPU(0)" << std::endl;
    out << "super = src.KFMSuper(src.KFMPad()).OnCPU(0)" << std::endl;
    out << "clip60 = src.Bob().OnCPU(0)" << std::endl;
    out << "fmclip = super.KPreCycleAnalyze().KFMCycleAnalyze(src).OnCPU(0)" << std::endl;
    out << "clip24 = src.KTelecine(fmclip).OnCPU(0)" << std::endl;
    out << "flag = super.KSwitchFlag()" << std::endl;
    out << "combemask = src.KCombeMask(flag).OnCPU(0)" << std::endl;
    out << "containscombe = flag.KContainsCombe().OnCPU(0)" << std::endl;

    out << "ref = clip24.KPatchCombe(clip60, fmclip, combemask, containscombe)" << std::endl;
    out << "cuda = clip24.KPatchCombe(clip60, fmclip, combemask, containscombe)" O_C(0) "" << std::endl;

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

TEST_F(KFMTest, SwitchTest)
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

    out << "src = LWLibavVideoSource(\"test.ts\").OnCPU(0)" << std::endl;
    out << "super = src.KFMSuper(src.KFMPad()).OnCPU(0)" << std::endl;
    out << "clip60 = src.Bob().OnCPU(0)" << std::endl;
    out << "fmclip = super.KPreCycleAnalyze().KFMCycleAnalyze(src).OnCPU(0)" << std::endl;
    out << "clip24 = src.KTelecine(fmclip).OnCPU(0)" << std::endl;
    out << "super24 = super.KTelecineSuper(fmclip).OnCPU(0)" << std::endl;
    out << "flag24 = super24.KSwitchFlag()" << std::endl;
    out << "mask24 = src.KCombeMask(flag24).OnCPU(0)" << std::endl;
    out << "cc24 = flag24.KContainsCombe().OnCPU(0)" << std::endl;
    out << "super30 = super.SelectEven().OnCPU(0)" << std::endl;
    out << "flag30 = super30.KSwitchFlag()" << std::endl;
    out << "mask30 = src.KCombeMask(flag30).OnCPU(0)" << std::endl;
    out << "cc30 = flag30.KContainsCombe().OnCPU(0)" << std::endl;

    out << "ref = clip60.KFMSwitch(fmclip, clip24, mask24, cc24, src, mask30, cc30, thswitch=40)" << std::endl;
    out << "cuda = clip60.KFMSwitch(fmclip, clip24, mask24, cc24, src, mask30, cc30, thswitch=40)" O_C(0) "" << std::endl;

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

void KFMTest::TemporalNRTest(TEST_FRAMES tf)
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
    out << "srcuda = src.ConvertBits(14).OnCPU(0)" << std::endl;

    out << "ref = src.KTemporalNR(3, 4).ConvertBits(8)" << std::endl;
    out << "cuda = srcuda.KTemporalNR(3, 4)" O_C(0) ".ConvertBits(8)" << std::endl;

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

TEST_F(KFMTest, TemporalNRTest)
{
  TemporalNRTest(TF_MID);
}

#pragma endregion

#pragma region Deband

void KFMTest::DebandTest(TEST_FRAMES tf, int sample_mode, bool blur_first)
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

TEST_F(KFMTest, DebandTest_Mode0F)
{
  DebandTest(TF_MID, 0, false);
}

TEST_F(KFMTest, DebandTest_Mode1F)
{
  DebandTest(TF_MID, 1, false);
}

TEST_F(KFMTest, DebandTest_Mode2F)
{
  DebandTest(TF_MID, 2, false);
}

TEST_F(KFMTest, DebandTest_Mode0T)
{
  DebandTest(TF_MID, 0, true);
}

TEST_F(KFMTest, DebandTest_Mode1T)
{
  DebandTest(TF_MID, 1, true);
}

TEST_F(KFMTest, DebandTest_Mode2T)
{
  DebandTest(TF_MID, 2, true);
}

#pragma endregion

#pragma region EdgeLevel

void KFMTest::EdgeLevelTest(TEST_FRAMES tf, int repair, bool chroma)
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
		if (chroma) {
			out << "src = src.ConvertToYUV444()" << std::endl;
		}
    out << "srcuda = src.OnCPU(0)" << std::endl;

    out << "ref = src.KEdgeLevel(16, 10, " << repair << ", strUV=" << (chroma ? 32 : 0) << ")" << std::endl;
    out << "cuda = srcuda.KEdgeLevel(16, 10, " << repair << ", strUV=" << (chroma ? 32 : 0) << ")" O_C(0) "" << std::endl;

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

TEST_F(KFMTest, EdgeLevel_Rep0WithC)
{
  EdgeLevelTest(TF_MID, 0, true);
}

TEST_F(KFMTest, EdgeLevel_Rep1WithC)
{
  EdgeLevelTest(TF_MID, 1, true);
}

TEST_F(KFMTest, EdgeLevel_Rep2WithC)
{
  EdgeLevelTest(TF_MID, 2, true);
}

TEST_F(KFMTest, EdgeLevel_Rep0NoC)
{
  EdgeLevelTest(TF_MID, 0, false);
}

TEST_F(KFMTest, EdgeLevel_Rep1NoC)
{
  EdgeLevelTest(TF_MID, 1, false);
}

TEST_F(KFMTest, EdgeLevel_Rep2NoC)
{
  EdgeLevelTest(TF_MID, 2, false);
}

#pragma endregion

#pragma region CFieldDiff

void KFMTest::CFieldDiffTest(int nt, bool chroma)
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

    out << "global current_frame = 100" << std::endl;
    out << "ref = src.CFieldDiff(nt = " << nt << ", chroma=" << (chroma ? "true" : "false") << ")" << std::endl;
    out << "cuda = OnCUDA(function[srcuda](){srcuda.KCFieldDiff(nt = "
      << nt << ", chroma=" << (chroma ? "true" : "false") << ")})" << std::endl;

    out.close();

    {
      env->Invoke("Import", scriptpath.c_str());
      double ref = env->GetVar("ref").AsFloat();
      double cuda = env->GetVar("cuda").AsFloat();
      // 境界の扱いが異なるので（多分）一致しない
      // 差が1%未満であることを確認
      if (std::abs(ref - cuda) / ref >= 0.02) {
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

TEST_F(KFMTest, CFieldDiff_Nt0WithC)
{
  CFieldDiffTest(0, true);
}

TEST_F(KFMTest, CFieldDiff_Nt0NoC)
{
  CFieldDiffTest(0, false);
}

TEST_F(KFMTest, CFieldDiff_Nt3WithC)
{
  CFieldDiffTest(3, true);
}

TEST_F(KFMTest, CFieldDiff_Nt3NoC)
{
  CFieldDiffTest(3, false);
}

#pragma endregion

#pragma region CFrameDiffDup

void KFMTest::CFrameDiffDupTest(int blocksize, bool chroma)
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

    out << "global current_frame = 100" << std::endl;
    out << "ref = src.KCFrameDiffDup(blksize = " << blocksize << ", chroma=" << (chroma ? "true" : "false") << ")" << std::endl;
    out << "cuda = OnCUDA(function[srcuda](){srcuda.KCFrameDiffDup(blksize = "
      << blocksize << ", chroma=" << (chroma ? "true" : "false") << ")})" << std::endl;

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

TEST_F(KFMTest, CFrameDiffDup_Blk32WithC)
{
  CFrameDiffDupTest(32, true);
}

TEST_F(KFMTest, CFrameDiffDup_Blk32NoC)
{
  CFrameDiffDupTest(32, false);
}

TEST_F(KFMTest, CFrameDiffDup_Blk8WithC)
{
  CFrameDiffDupTest(8, true);
}

TEST_F(KFMTest, CFrameDiffDup_Blk8NoC)
{
  CFrameDiffDupTest(8, false);
}

#pragma endregion

#pragma region DecombUCF

void KFMTest::DecombUCF24Test(TEST_FRAMES tf, int chroma, bool show)
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

    out << "src = LWLibavVideoSource(\"test.ts\").OnCPU(0)" << std::endl;
    out << "fields = src.SeparateFields()" << std::endl;
    out << "bob = src.Bob().OnCPU(0)" << std::endl;
    out << "noise = fields.GaussResize(1920,540,0,0,1920.0001,540.0001,p=2).Crop(4,4,-4,-4).Align().OnCPU(0)" << std::endl;
    out << "noise = src.KAnalyzeNoise(fields.Crop(4,4,-4,-4).Align().KNoiseClip(noise), src.KFMSuper())" << std::endl;

    out << "ref = src.KDecombUCF(noise, bob, chroma=" << chroma << (show ? ", show=true" : "") << ")" << std::endl;
    out << "cuda = src.KDecombUCF(noise" O_C(0) ", bob, chroma=" << chroma << (show ? ", show=true" : "") << ")" O_C(0) << std::endl;

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

TEST_F(KFMTest, DecombUCF_C0NoShow)
{
  DecombUCF24Test(TF_MID, 0, false);
}

TEST_F(KFMTest, DecombUCF_C1NoShow)
{
  DecombUCF24Test(TF_MID, 1, false);
}

TEST_F(KFMTest, DecombUCF_C2NoShow)
{
  DecombUCF24Test(TF_MID, 2, false);
}

TEST_F(KFMTest, DecombUCF_C1Show)
{
  DecombUCF24Test(TF_MID, 1, true);
}

#pragma endregion

#pragma region Deblock

void KFMTest::DeblockTest(TEST_FRAMES tf, int quality, bool is_soft)
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

    out << "src = LWLibavVideoSource(\"test.ts\").OnCPU(0)" << std::endl;
    out << "ref = src.KDeblock(quality=" << quality << ",force_qp=10,is_soft=" << (is_soft ? "true" : "false") << ")" << std::endl;
    out << "cuda = src.KDeblock(quality=" << quality << ",force_qp=10,is_soft=" << (is_soft ? "true" : "false") << ")" O_C(0) << std::endl;

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

TEST_F(KFMTest, Deblock_Hard)
{
  printf("Deblock Hard Quality 1\n");
  DeblockTest(TF_MID, 1, false);
  printf("Deblock Hard Quality 2\n");
  DeblockTest(TF_MID, 2, false);
  printf("Deblock Hard Quality 3\n");
  DeblockTest(TF_MID, 3, false);
  printf("Deblock Hard Quality 4\n");
  DeblockTest(TF_MID, 4, false);
  printf("Deblock Hard Quality 5\n");
  DeblockTest(TF_MID, 5, false);
  printf("Deblock Hard Quality 6\n");
  DeblockTest(TF_MID, 6, false);
}

TEST_F(KFMTest, Deblock_Soft)
{
  printf("Deblock Soft Quality 1\n");
  DeblockTest(TF_MID, 1, true);
  printf("Deblock Soft Quality 2\n");
  DeblockTest(TF_MID, 2, true);
  printf("Deblock Soft Quality 3\n");
  DeblockTest(TF_MID, 3, true);
  printf("Deblock Soft Quality 4\n");
  DeblockTest(TF_MID, 4, true);
  printf("Deblock Soft Quality 5\n");
  DeblockTest(TF_MID, 5, true);
  printf("Deblock Soft Quality 6\n");
  DeblockTest(TF_MID, 6, true);
}

#pragma endregion
