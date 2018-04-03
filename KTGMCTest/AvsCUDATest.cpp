
#include "TestCommons.h"

class AvsCUDATest : public AvsTestBase {
protected:
  AvsCUDATest() { }

  enum FORMAT {
    FORMAT_YUV, FORMAT_RGB, FORMAT_PLANAR_RGB
  };

  void CondTest_(const char* fname, bool is_cuda, FORMAT format, bool two_arg);
  void CondTest(const char* fname, FORMAT format, bool two_arg = false);
};

void AvsCUDATest::CondTest_(const char* fname, bool is_cuda, FORMAT format, bool two_arg)
{
  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string ktgmcPath = modulePath + "\\AvsCUDA.dll";
    env->LoadPlugin(ktgmcPath.c_str(), true, &result);

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"test.ts\")" << std::endl;

    if (format == FORMAT_RGB) {
      out << "src = src.ConvertToRGB()" << std::endl;
    }
    else if (format == FORMAT_PLANAR_RGB) {
      out << "src = src.ConvertToPlanarRGB()" << std::endl;
    }

    if (two_arg) {
      out << "src2 = src.GaussResize(1920,1080,0,0,1920.0001,1080.0001,p=2)" << std::endl;
    }

    out << "current_frame = 100" << std::endl;
    out << "ref = src." << fname << "(" << (two_arg ? "src2" : "") << ")" << std::endl;

    if (is_cuda) {
      out << "srcuda = src.OnCPU(0)" << std::endl;
      if (two_arg) {
        out << "srcuda2 = src2.OnCPU(0)" << std::endl;
      }
      out << "cuda = EvalOnCUDA(\"srcuda.K" << fname << "(" << (two_arg ? "srcuda2" : "") << ")\")" << std::endl;
    }
    else {
      out << "cuda = src.K" << fname << "(" << (two_arg ? "src2" : "") << ")" << std::endl;
    }

    out.close();

    {
      env->Invoke("Import", scriptpath.c_str());
      double ref = env->GetVar("ref").AsFloat();
      double cuda = env->GetVar("cuda").AsFloat();
      if (ref != cuda) {
        printf("’l‚ªˆá‚¢‚Ü‚· %f vs %f\n", ref, cuda);
        GTEST_FAIL();
      }
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

void AvsCUDATest::CondTest(const char* fname, FORMAT format, bool two_arg)
{
  CondTest_(fname, false, format, two_arg);
  CondTest_(fname, true, format, two_arg);
}


TEST_F(AvsCUDATest, AverageLuma)
{
  CondTest("AverageLuma", FORMAT_YUV);
}

TEST_F(AvsCUDATest, AverageChromaU)
{
  CondTest("AverageChromaU", FORMAT_YUV);
}

TEST_F(AvsCUDATest, AverageChromaV)
{
  CondTest("AverageChromaV", FORMAT_YUV);
}

TEST_F(AvsCUDATest, AverageR)
{
  CondTest("AverageR", FORMAT_PLANAR_RGB);
}

TEST_F(AvsCUDATest, AverageG)
{
  CondTest("AverageG", FORMAT_PLANAR_RGB);
}

TEST_F(AvsCUDATest, AverageB)
{
  CondTest("AverageB", FORMAT_PLANAR_RGB);
}

TEST_F(AvsCUDATest, RGBDifference)
{
  CondTest("RGBDifference", FORMAT_RGB, true);
}

TEST_F(AvsCUDATest, LumaDifference)
{
  CondTest("LumaDifference", FORMAT_YUV, true);
}

TEST_F(AvsCUDATest, ChromaUDifference)
{
  CondTest("ChromaUDifference", FORMAT_YUV, true);
}

TEST_F(AvsCUDATest, ChromaVDifference)
{
  CondTest("ChromaVDifference", FORMAT_YUV, true);
}

TEST_F(AvsCUDATest, RDifference)
{
  CondTest("RDifference", FORMAT_PLANAR_RGB, true);
}

TEST_F(AvsCUDATest, GDifference)
{
  CondTest("GDifference", FORMAT_PLANAR_RGB, true);
}

TEST_F(AvsCUDATest, BDifference)
{
  CondTest("BDifference", FORMAT_PLANAR_RGB, true);
}

TEST_F(AvsCUDATest, YDifferenceFromPrevious)
{
  CondTest("YDifferenceFromPrevious", FORMAT_YUV);
}

TEST_F(AvsCUDATest, UDifferenceFromPrevious)
{
  CondTest("UDifferenceFromPrevious", FORMAT_YUV);
}

TEST_F(AvsCUDATest, VDifferenceFromPrevious)
{
  CondTest("VDifferenceFromPrevious", FORMAT_YUV);
}

TEST_F(AvsCUDATest, RGBDifferenceFromPrevious)
{
  CondTest("RGBDifferenceFromPrevious", FORMAT_RGB);
}

TEST_F(AvsCUDATest, RDifferenceFromPrevious)
{
  CondTest("RDifferenceFromPrevious", FORMAT_PLANAR_RGB);
}

TEST_F(AvsCUDATest, GDifferenceFromPrevious)
{
  CondTest("GDifferenceFromPrevious", FORMAT_PLANAR_RGB);
}

TEST_F(AvsCUDATest, BDifferenceFromPrevious)
{
  CondTest("BDifferenceFromPrevious", FORMAT_PLANAR_RGB);
}

TEST_F(AvsCUDATest, YDifferenceToNext)
{
  CondTest("YDifferenceToNext", FORMAT_YUV);
}

TEST_F(AvsCUDATest, UDifferenceToNext)
{
  CondTest("UDifferenceToNext", FORMAT_YUV);
}

TEST_F(AvsCUDATest, VDifferenceToNext)
{
  CondTest("VDifferenceToNext", FORMAT_YUV);
}

TEST_F(AvsCUDATest, RGBDifferenceToNext)
{
  CondTest("RGBDifferenceToNext", FORMAT_RGB);
}

TEST_F(AvsCUDATest, RDifferenceToNext)
{
  CondTest("RDifferenceToNext", FORMAT_PLANAR_RGB);
}

TEST_F(AvsCUDATest, GDifferenceToNext)
{
  CondTest("GDifferenceToNext", FORMAT_PLANAR_RGB);
}

TEST_F(AvsCUDATest, BDifferenceToNext)
{
  CondTest("BDifferenceToNext", FORMAT_PLANAR_RGB);
}

TEST_F(AvsCUDATest, YPlaneMax)
{
  CondTest("YPlaneMax", FORMAT_YUV);
}

TEST_F(AvsCUDATest, YPlaneMin)
{
  CondTest("YPlaneMin", FORMAT_YUV);
}

TEST_F(AvsCUDATest, YPlaneMedian)
{
  CondTest("YPlaneMedian", FORMAT_YUV);
}

TEST_F(AvsCUDATest, UPlaneMax)
{
  CondTest("UPlaneMax", FORMAT_YUV);
}

TEST_F(AvsCUDATest, UPlaneMin)
{
  CondTest("UPlaneMin", FORMAT_YUV);
}

TEST_F(AvsCUDATest, UPlaneMedian)
{
  CondTest("UPlaneMedian", FORMAT_YUV);
}

TEST_F(AvsCUDATest, VPlaneMax)
{
  CondTest("VPlaneMax", FORMAT_YUV);
}

TEST_F(AvsCUDATest, VPlaneMin)
{
  CondTest("VPlaneMin", FORMAT_YUV);
}

TEST_F(AvsCUDATest, VPlaneMedian)
{
  CondTest("VPlaneMedian", FORMAT_YUV);
}

TEST_F(AvsCUDATest, RPlaneMax)
{
  CondTest("RPlaneMax", FORMAT_PLANAR_RGB);
}

TEST_F(AvsCUDATest, RPlaneMin)
{
  CondTest("RPlaneMin", FORMAT_PLANAR_RGB);
}

TEST_F(AvsCUDATest, RPlaneMedian)
{
  CondTest("RPlaneMedian", FORMAT_PLANAR_RGB);
}

TEST_F(AvsCUDATest, GPlaneMax)
{
  CondTest("GPlaneMax", FORMAT_PLANAR_RGB);
}

TEST_F(AvsCUDATest, GPlaneMin)
{
  CondTest("GPlaneMin", FORMAT_PLANAR_RGB);
}

TEST_F(AvsCUDATest, GPlaneMedian)
{
  CondTest("GPlaneMedian", FORMAT_PLANAR_RGB);
}

TEST_F(AvsCUDATest, BPlaneMax)
{
  CondTest("BPlaneMax", FORMAT_PLANAR_RGB);
}

TEST_F(AvsCUDATest, BPlaneMin)
{
  CondTest("BPlaneMin", FORMAT_PLANAR_RGB);
}

TEST_F(AvsCUDATest, BPlaneMedian)
{
  CondTest("BPlaneMedian", FORMAT_PLANAR_RGB);
}

TEST_F(AvsCUDATest, YPlaneMinMaxDifference)
{
  CondTest("YPlaneMinMaxDifference", FORMAT_YUV);
}

TEST_F(AvsCUDATest, UPlaneMinMaxDifference)
{
  CondTest("UPlaneMinMaxDifference", FORMAT_YUV);
}

TEST_F(AvsCUDATest, VPlaneMinMaxDifference)
{
  CondTest("VPlaneMinMaxDifference", FORMAT_YUV);
}

TEST_F(AvsCUDATest, RPlaneMinMaxDifference)
{
  CondTest("RPlaneMinMaxDifference", FORMAT_PLANAR_RGB);
}

TEST_F(AvsCUDATest, GPlaneMinMaxDifference)
{
  CondTest("GPlaneMinMaxDifference", FORMAT_PLANAR_RGB);
}

TEST_F(AvsCUDATest, BPlaneMinMaxDifference)
{
  CondTest("BPlaneMinMaxDifference", FORMAT_PLANAR_RGB);
}
