
#include "TestCommons.h"

class MaskToolsTest : public AvsTestBase {
protected:
   MaskToolsTest() { }
};


TEST_F(MaskToolsTest, Lut_1)
{
   PEnv env;
   try {
      env = PEnv(CreateScriptEnvironment2());

      AVSValue result;
      std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
      env->LoadPlugin(debugtoolPath.c_str(), true, &result);
      std::string ktgmcPath = modulePath + "\\KMaskTools.dll";
      env->LoadPlugin(ktgmcPath.c_str(), true, &result);

      std::string scriptpath = workDirPath + "\\script.avs";

      std::ofstream out(scriptpath);

      const char* expry = "\"0.000000 1.062500 0.066406 x 16 - 219 / 0 1 clip 0.062500 + / -*x 16 - 219 / 0 1 clip 1 0.000000 - *+255 * \"";
      const char* expr = "\"x 128 - 128 * 112 / 128 + \"";

      out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
      out << "ref = src.mt_lut(yexpr=" << expry << ",expr=" << expr << ",y=3,u=3,v=3)" << std::endl;
      out << "cuda = src.OnCPU(0).kmt_lut(yexpr=" << expry << ",expr=" << expr << ",y=3,u=3,v=3).OnCUDA(0)" << std::endl;
      out << "ImageCompare(ref, cuda, 1, true)" << std::endl;

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

TEST_F(MaskToolsTest, Lutxy_1)
{
   PEnv env;
   try {
      env = PEnv(CreateScriptEnvironment2());

      AVSValue result;
      std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
      env->LoadPlugin(debugtoolPath.c_str(), true, &result);
      std::string ktgmcPath = modulePath + "\\KMaskTools.dll";
      env->LoadPlugin(ktgmcPath.c_str(), true, &result);

      std::string scriptpath = workDirPath + "\\script.avs";

      std::ofstream out(scriptpath);

      const char* expr = "\"clamp_f x x y - 0.700000 * +\"";

      out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;
      out << "sref = src.GaussResize(1920,1080,0,0,1920.0001,1080.0001,p=2)" << std::endl;
      out << "srcuda = src.OnCPU(0)" << std::endl;
      out << "srefcuda = sref.OnCPU(0)" << std::endl;

      out << "ref = src.mt_lutxy(sref," << expr << ",U=3,V=3)" << std::endl;
      out << "cuda = srcuda.kmt_lutxy(srefcuda,"<< expr<<",U=3,V=3).OnCUDA(0)" << std::endl;
      out << "ImageCompare(ref, cuda, 1, true)" << std::endl;

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

TEST_F(MaskToolsTest, Func)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "LWLibavVideoSource(\"test.ts\")" << std::endl;

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

