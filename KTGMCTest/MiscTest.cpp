
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
    out << "fields = SeparateFields().Crop(4,4,-4,-4).Align()" << std::endl;
    out << "bob = KTGMC_Bob()" << std::endl;
    out << "noise = fields.KGaussResize(p=2.5)" << std::endl;
    out << "noise = last.KAnalyzeNoise(fields.KNoiseClip(noise), KFMSuper()).OnCUDA(2)" << std::endl;
    out << "bob.KDecombUCF60(noise, bob, bob).OnCUDA(0)" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      for (int i = 100; i < 1100; ++i) {
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

TEST_F(MiscTest, KFMPerf)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\").OnCPU(0)" << std::endl;
    out << "super = src.KFMSuper(src.KFMPad())" << std::endl;
    out << "clip60 = src.KTGMC_Bob()" << std::endl;
    out << "fmclip = super.KPreCycleAnalyze().OnCUDA(0).KFMCycleAnalyze(src).OnCPU(0)" << std::endl;
    out << "clip24 = src.KTelecine(fmclip)" << std::endl;
    out << "mask24 = src.KCombeMask(super.KTelecineSuper(fmclip).KSwitchFlag())" << std::endl;
    out << "cc24 = mask24.OnCUDA(0).KContainsCombe()" << std::endl;
    out << "mask30 = src.KCombeMask(super.SelectEven().KSwitchFlag())" << std::endl;
    out << "cc30 = mask30.OnCUDA(0).KContainsCombe()" << std::endl;
    out << "clip60.KFMSwitch(fmclip, clip24, mask24, cc24, src, mask30, cc30, thswitch=40).OnCUDA(0)" << std::endl;

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

