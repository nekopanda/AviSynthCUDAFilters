
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

  void DecombeUCFTest(TEST_FRAMES tf, int chroma, bool show);
  void DecombeUCF24Test(TEST_FRAMES tf, int chroma, bool show);
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

TEST_F(KFMTest, AnalyzeFrameTest)
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

TEST_F(KFMTest, AnalyzeFrameSuperTest)
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

    out << "ref = src.KFMFrameAnalyze()" << std::endl;
    out << "cuda = src.KFMFrameAnalyze(super=super)" O_C(0) "" << std::endl;

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

#pragma region DecombeUCF

void KFMTest::DecombeUCF24Test(TEST_FRAMES tf, int chroma, bool show)
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

    out << "ref = src.KDecombeUCF24(noise, bob, chroma=" << chroma << (show ? ", show=true" : "") << ")" << std::endl;
    out << "cuda = src.KDecombeUCF24(noise" O_C(0) ", bob, chroma=" << chroma << (show ? ", show=true" : "") << ")" O_C(0) << std::endl;

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

TEST_F(KFMTest, DecombeUCF_C0NoShow)
{
  DecombeUCF24Test(TF_MID, 0, false);
}

TEST_F(KFMTest, DecombeUCF_C1NoShow)
{
  DecombeUCF24Test(TF_MID, 1, false);
}

TEST_F(KFMTest, DecombeUCF_C2NoShow)
{
  DecombeUCF24Test(TF_MID, 2, false);
}

TEST_F(KFMTest, DecombeUCF_C1Show)
{
  DecombeUCF24Test(TF_MID, 1, true);
}

#pragma endregion

