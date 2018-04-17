
#include "TestCommons.h"

class MiscTest : public AvsTestBase {
protected:
  MiscTest() { }
};

TEST_F(MiscTest, UCFPerf)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "Import(\"30_ucf.avs\")" << std::endl;
    out << "LWLibavVideoSource(\"test.ts\")" << std::endl;
    out << "kgauss = function(clip c) {c.Align().KGaussResize(p=2.5,chroma=true)}" << std::endl;
    out << "OnCPU(2).DecombUCF2(bob_clip=D3DVP().OnCPU(2),affect_noise=kgauss,cuda=true)" << std::endl;

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

TEST_F(MiscTest, UCF2Perf)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "LWLibavVideoSource(\"test.ts\").OnCPU(1)" << std::endl;
    out << "fields = SeparateFields().Align()" << std::endl;
    out << "bob = KTGMC_Bob()" << std::endl;
    out << "noise = fields.KGaussResize(p=2.5).Crop(4,4,-4,-4).Align()" << std::endl;
    out << "noise = KAnalyzeNoise(fields.Crop(4,4,-4,-4).Align().KNoiseClip(noise), KFMSuper()).OnCUDA(1)" << std::endl;
    out << "KDecombeUCF24(noise, bob, chroma=1, show=true).OnCUDA(1)" << std::endl;

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

TEST_F(MiscTest, KFMPerf)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\",dominance=1,repeat=True).OnCPU(2)" << std::endl;
    out << "fm = src.KFMFrameAnalyze(5, 4, 7, 6)" << std::endl;
    out << "fm = fm.OnCUDA(0).KFMCycleAnalyze(src, 5, 1).OnCPU(2)" << std::endl;
    out << "tc = src.KTelecine(fm)" << std::endl;
    out << "tc.KRemoveCombe(6, 100).OnCUDA(2)" << std::endl;

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

