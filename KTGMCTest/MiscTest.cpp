
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

    out << "SetMemoryMax(1500, type=DEV_TYPE_CUDA)" << std::endl;
    out << "SetDeviceOpt(DEV_CUDA_PINNED_HOST)" << std::endl;
    out << "Import(\"KFMDeint.avs\")" << std::endl;
    out << "src = LWLibavVideoSource(\"test.ts\").OnCPU(0)" << std::endl;
    out << "src.KFMDeint(mode=0, ucf=true, nr=true, cuda=true, show=true)" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      GetFrames(clip, TF_100, env.get());
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}


TEST_F(MiscTest, GenericScriptTest)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "Import(\"T:\\sandbox\\t28\\AvsTest\\63_ktgmc_perf.avs\")" << std::endl;

    out.close();

    {
      PClip clip = env->Invoke("Import", scriptpath.c_str()).AsClip();
      GetFrames(clip, TF_100, env.get());
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

#include "../KFM/SIMDSupport.hpp"

TEST_F(MiscTest, SIMDSupportTest)
{
	__m256i t = const_mm256_setr_epi16(
		(int16_t)0x1234, (int16_t)0x2345, (int16_t)0x3456, (int16_t)0x4567,
		(int16_t)0x5678, (int16_t)0x6789, (int16_t)0x789A, (int16_t)0x89AB,
		(int16_t)0x9ABC, (int16_t)0xABCD, (int16_t)0xBCDE, (int16_t)0xCDEF,
		(int16_t)0xDEF0, (int16_t)0xEF01, (int16_t)0xF012, (int16_t)0x0123);
	__m256i r = _mm256_setr_epi16(
		(int16_t)0x1234, (int16_t)0x2345, (int16_t)0x3456, (int16_t)0x4567,
		(int16_t)0x5678, (int16_t)0x6789, (int16_t)0x789A, (int16_t)0x89AB,
		(int16_t)0x9ABC, (int16_t)0xABCD, (int16_t)0xBCDE, (int16_t)0xCDEF,
		(int16_t)0xDEF0, (int16_t)0xEF01, (int16_t)0xF012, (int16_t)0x0123);
	EXPECT_EQ(memcmp(&t, &r, sizeof(t)), 0);
}
