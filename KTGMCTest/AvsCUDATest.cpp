
#include "TestCommons.h"

class AvsCUDATest : public AvsTestBase {
protected:
  AvsCUDATest() { }

  enum FORMAT {
    FORMAT_YV420,
    //FORMAT_YV411, // not support
    FORMAT_YV422,
    FORMAT_YV444,
    FORMAT_YUY2,
    FORMAT_Y,
    FORMAT_RGB,
    FORMAT_RGBA,
    FORMAT_PLANAR_RGB,
    FORMAT_PLANAR_RGBA
  };

  void ConvertFormat(std::ofstream& out, FORMAT format, int bits);

  static bool IsRGB(FORMAT format) {
    return (format == FORMAT_RGB) || (format == FORMAT_RGBA) ||
      (format == FORMAT_PLANAR_RGB) || (format == FORMAT_PLANAR_RGBA);
  }
  static const char* FormatString(FORMAT format) {
    switch (format) {
    case FORMAT_YV420: return "FORMAT_YV420";
    //case FORMAT_YV411: return "FORMAT_YV411";
    case FORMAT_YV422: return "FORMAT_YV422";
    case FORMAT_YV444: return "FORMAT_YV444";
    case FORMAT_YUY2: return "FORMAT_YUY2";
    case FORMAT_Y: return "FORMAT_Y";
    case FORMAT_RGB: return "FORMAT_RGB";
    case FORMAT_RGBA: return "FORMAT_RGBA";
    case FORMAT_PLANAR_RGB: return "FORMAT_PLANAR_RGB";
    case FORMAT_PLANAR_RGBA: return "FORMAT_PLANAR_RGBA";
    }
    return "Unknown format";
  }
};

void AvsCUDATest::ConvertFormat(std::ofstream& out, FORMAT format, int bits)
{
  if (bits != 8) {
    out << "src = src.ConvertBits(" << bits << ")" << std::endl;
  }

  switch (format) {
  //case FORMAT_YV411:
  //  out << "src = src.ConvertToYV411(interlaced=true)" << std::endl;
  //  break;
  case FORMAT_YV422:
    out << "src = src.ConvertToYUV422(interlaced=true)" << std::endl;
    break;
  case FORMAT_YV444:
    out << "src = src.ConvertToYUV444(interlaced=true)" << std::endl;
    break;
  case FORMAT_YUY2:
    out << "src = src.ConvertToYUY2(interlaced=true)" << std::endl;
    break;
  case FORMAT_Y:
    out << "src = src.ConvertToY()" << std::endl;
    break;
  case FORMAT_RGB:
    if (bits == 8) {
      out << "src = src.ConvertToRGB24(interlaced=true)" << std::endl;
    }
    else {
      out << "src = src.ConvertToRGB48(interlaced=true)" << std::endl;
    }
    break;
  case FORMAT_RGBA:
    if (bits == 8) {
      out << "src = src.ConvertToRGB32(interlaced=true)" << std::endl;
    }
    else {
      out << "src = src.ConvertToRGB64(interlaced=true)" << std::endl;
    }
    break;
  case FORMAT_PLANAR_RGB:
    out << "src = src.ConvertToPlanarRGB(interlaced=true)" << std::endl;
    break;
  case FORMAT_PLANAR_RGBA:
    out << "src = src.ConvertToPlanarRGBA(interlaced=true)" << std::endl;
    break;
  }
}

#pragma region CondFunc

class CondFuncTest : public AvsCUDATest
{
protected:
  void CondTest_(const char* fname, bool is_cuda, FORMAT format, int bits, bool two_arg);
  void CondTest(const char* fname, std::vector<FORMAT> formats, bool two_arg = false);
};

void CondFuncTest::CondTest_(const char* fname, bool is_cuda, FORMAT format, int bits, bool two_arg)
{
  printf("%s(%s): %s %d bits\n", fname, is_cuda ? "CUDA" : "CPU ", FormatString(format), bits);

  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string pluginPath = modulePath + "\\AvsCUDA.dll";

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"news.ts\")" << std::endl;

    ConvertFormat(out, format, bits);

    if (two_arg) {
      out << "src2 = src.GaussResize(1920,1080,0,0,1920.0001,1080.0001,p=2)" << std::endl;
    }

    out << "global current_frame = 100" << std::endl;
    out << "ref = src." << fname << "(" << (two_arg ? "src2" : "") << ")" << std::endl;

    out << "LoadPlugin(\"" << pluginPath.c_str() << "\")" << std::endl;
    if (is_cuda) {
      out << "srcuda = src.OnCPU(0)" << std::endl;
      if (two_arg) {
        out << "srcuda2 = src2.OnCPU(0)" << std::endl;
      }
      out << "cuda = OnCUDA(function[srcuda" <<  (two_arg ? ", srcuda2" : "") << "](){srcuda." 
        << fname << "(" << (two_arg ? "srcuda2" : "") << ")})" << std::endl;
    }
    else {
      out << "cuda = src." << fname << "(" << (two_arg ? "src2" : "") << ")" << std::endl;
    }

    out.close();

    {
      env->Invoke("Import", scriptpath.c_str());
      double ref = env->GetVar("ref").AsFloat();
      double cuda = env->GetVar("cuda").AsFloat();
      if (bits == 32) {
        // 有効数字が分からないので適当に8bit合ってればOKにする
        if (std::abs(ref - cuda) / std::abs(ref) >= std::abs(ref) / (1 << 8)) {
          printf("値が違います(float) %f vs %f\n", ref, cuda);
          GTEST_FAIL();
        }
      }
      else {
        if (ref != cuda) {
          printf("値が違います %f vs %f\n", ref, cuda);
          GTEST_FAIL();
        }
      }
    }
  }
  catch (const AvisynthError& err) {
    printf("%s\n", err.msg);
    GTEST_FAIL();
  }
}

void CondFuncTest::CondTest(const char* fname, std::vector<FORMAT> formats, bool two_arg)
{
  int bits[] = { 8, 16, 10, 12, 14, 32 };
  for (int i = 0; i < (int)formats.size(); ++i) {
    int nbits = IsRGB(formats[i]) ? 2 : 6;
    for (int b = 0; b < nbits; ++b) {
      CondTest_(fname, false, formats[i], bits[b], two_arg);
      CondTest_(fname, true, formats[i], bits[b], two_arg);
    }
  }
}


TEST_F(CondFuncTest, AverageLuma)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y };
  CondTest("AverageLuma", formats);
}

TEST_F(CondFuncTest, AverageChromaU)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444 };
  CondTest("AverageChromaU", formats);
}

TEST_F(CondFuncTest, AverageChromaV)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444 };
  CondTest("AverageChromaV", formats);
}

TEST_F(CondFuncTest, AverageR)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("AverageR", formats);
}

TEST_F(CondFuncTest, AverageG)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("AverageG", formats);
}

TEST_F(CondFuncTest, AverageB)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("AverageB", formats);
}

TEST_F(CondFuncTest, RGBDifference)
{
  std::vector<FORMAT> formats = { FORMAT_RGB, FORMAT_RGBA };
  CondTest("RGBDifference", formats, true);
}

TEST_F(CondFuncTest, LumaDifference)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y };
  CondTest("LumaDifference", formats, true);
}

TEST_F(CondFuncTest, ChromaUDifference)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444 };
  CondTest("ChromaUDifference", formats, true);
}

TEST_F(CondFuncTest, ChromaVDifference)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444 };
  CondTest("ChromaVDifference", formats, true);
}

TEST_F(CondFuncTest, RDifference)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("RDifference", formats, true);
}

TEST_F(CondFuncTest, GDifference)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("GDifference", formats, true);
}

TEST_F(CondFuncTest, BDifference)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("BDifference", formats, true);
}

TEST_F(CondFuncTest, YDifferenceFromPrevious)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y };
  CondTest("YDifferenceFromPrevious", formats);
}

TEST_F(CondFuncTest, UDifferenceFromPrevious)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444 };
  CondTest("UDifferenceFromPrevious", formats);
}

TEST_F(CondFuncTest, VDifferenceFromPrevious)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444 };
  CondTest("VDifferenceFromPrevious", formats);
}

TEST_F(CondFuncTest, RGBDifferenceFromPrevious)
{
  std::vector<FORMAT> formats = { FORMAT_RGB, FORMAT_RGBA };
  CondTest("RGBDifferenceFromPrevious", formats);
}

TEST_F(CondFuncTest, RDifferenceFromPrevious)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("RDifferenceFromPrevious", formats);
}

TEST_F(CondFuncTest, GDifferenceFromPrevious)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("GDifferenceFromPrevious", formats);
}

TEST_F(CondFuncTest, BDifferenceFromPrevious)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("BDifferenceFromPrevious", formats);
}

TEST_F(CondFuncTest, YDifferenceToNext)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y };
  CondTest("YDifferenceToNext", formats);
}

TEST_F(CondFuncTest, UDifferenceToNext)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444 };
  CondTest("UDifferenceToNext", formats);
}

TEST_F(CondFuncTest, VDifferenceToNext)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444 };
  CondTest("VDifferenceToNext", formats);
}

TEST_F(CondFuncTest, RGBDifferenceToNext)
{
  std::vector<FORMAT> formats = { FORMAT_RGB, FORMAT_RGBA };
  CondTest("RGBDifferenceToNext", formats);
}

TEST_F(CondFuncTest, RDifferenceToNext)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("RDifferenceToNext", formats);
}

TEST_F(CondFuncTest, GDifferenceToNext)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("GDifferenceToNext", formats);
}

TEST_F(CondFuncTest, BDifferenceToNext)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("BDifferenceToNext", formats);
}

TEST_F(CondFuncTest, YPlaneMax)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y };
  CondTest("YPlaneMax", formats);
}

TEST_F(CondFuncTest, YPlaneMin)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y };
  CondTest("YPlaneMin", formats);
}

TEST_F(CondFuncTest, YPlaneMedian)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y };
  CondTest("YPlaneMedian", formats);
}

TEST_F(CondFuncTest, UPlaneMax)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444 };
  CondTest("UPlaneMax", formats);
}

TEST_F(CondFuncTest, UPlaneMin)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444 };
  CondTest("UPlaneMin", formats);
}

TEST_F(CondFuncTest, UPlaneMedian)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444 };
  CondTest("UPlaneMedian", formats);
}

TEST_F(CondFuncTest, VPlaneMax)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444 };
  CondTest("VPlaneMax", formats);
}

TEST_F(CondFuncTest, VPlaneMin)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444 };
  CondTest("VPlaneMin", formats);
}

TEST_F(CondFuncTest, VPlaneMedian)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444 };
  CondTest("VPlaneMedian", formats);
}

TEST_F(CondFuncTest, RPlaneMax)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("RPlaneMax", formats);
}

TEST_F(CondFuncTest, RPlaneMin)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("RPlaneMin", formats);
}

TEST_F(CondFuncTest, RPlaneMedian)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("RPlaneMedian", formats);
}

TEST_F(CondFuncTest, GPlaneMax)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("GPlaneMax", formats);
}

TEST_F(CondFuncTest, GPlaneMin)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("GPlaneMin", formats);
}

TEST_F(CondFuncTest, GPlaneMedian)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("GPlaneMedian", formats);
}

TEST_F(CondFuncTest, BPlaneMax)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("BPlaneMax", formats);
}

TEST_F(CondFuncTest, BPlaneMin)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("BPlaneMin", formats);
}

TEST_F(CondFuncTest, BPlaneMedian)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("BPlaneMedian", formats);
}

TEST_F(CondFuncTest, YPlaneMinMaxDifference)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y };
  CondTest("YPlaneMinMaxDifference", formats);
}

TEST_F(CondFuncTest, UPlaneMinMaxDifference)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444 };
  CondTest("UPlaneMinMaxDifference", formats);
}

TEST_F(CondFuncTest, VPlaneMinMaxDifference)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444 };
  CondTest("VPlaneMinMaxDifference", formats);
}

TEST_F(CondFuncTest, RPlaneMinMaxDifference)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("RPlaneMinMaxDifference", formats);
}

TEST_F(CondFuncTest, GPlaneMinMaxDifference)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("GPlaneMinMaxDifference", formats);
}

TEST_F(CondFuncTest, BPlaneMinMaxDifference)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  CondTest("BPlaneMinMaxDifference", formats);
}

#pragma endregion

struct ScriptGen
{
  virtual void Pre(std::ofstream& out, const char* fname, bool is_cuda) const { }
  virtual void Ref(std::ofstream& out, const char* fname, bool is_cuda) const {
    out << "ref = src." << fname << std::endl;
  }
  virtual void Test(std::ofstream& out, const char* fname, bool is_cuda) const {
    if (is_cuda) {
      out << "srcuda = src.Align().OnCPU(0)" << std::endl;
      out << "cuda = srcuda." << fname << ".OnCUDA(0)" << std::endl;
    }
    else {
      out << "cuda = src." << fname << std::endl;
    }
  }
	virtual double Thresh(
		std::ofstream& out, const char* fname, bool is_cuda, int bits) const {
		return 1;
	}
};

class GenericTest : public AvsCUDATest
{
protected:
  void Test_(const char* fname, bool is_cuda, FORMAT format, int bits, const ScriptGen& gen);
  void Test(const char* fname, std::vector<FORMAT> formats, const ScriptGen& gen);
  void Test(const char* fname, std::vector<FORMAT> formats) {
    return Test(fname, formats, ScriptGen());
  }
};

void GenericTest::Test_(const char* fname, bool is_cuda, FORMAT format, int bits, const ScriptGen& gen)
{
  printf("%s%s: %s %d bits\n", fname, is_cuda ? "CUDA" : "CPU ", FormatString(format), bits);

  PEnv env;
  try {
    env = PEnv(CreateScriptEnvironment2());

    AVSValue result;
    std::string debugtoolPath = modulePath + "\\KDebugTool.dll";
    env->LoadPlugin(debugtoolPath.c_str(), true, &result);
    std::string pluginPath = modulePath + "\\AvsCUDA.dll";

    std::string scriptpath = workDirPath + "\\script.avs";

    std::ofstream out(scriptpath);

    out << "src = LWLibavVideoSource(\"news.ts\")" << std::endl;
    ConvertFormat(out, format, bits);
    gen.Pre(out, fname, is_cuda);
    gen.Ref(out, fname, is_cuda);
    out << "LoadPlugin(\"" << pluginPath.c_str() << "\")" << std::endl;
    gen.Test(out, fname, is_cuda);
    out << "ImageCompare(ref, cuda, "<< gen.Thresh(out, fname, is_cuda, bits) <<")" << std::endl;

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

void GenericTest::Test(const char* fname, std::vector<FORMAT> formats, const ScriptGen& gen)
{
  int bits[] = { 8, 16, 10, 12, 14, 32 };
  for (int i = 0; i < (int)formats.size(); ++i) {
    int nbits = IsRGB(formats[i]) ? 2 : 6;
    for (int b = 0; b < nbits; ++b) {
      Test_(fname, false, formats[i], bits[b], gen);
      Test_(fname, true, formats[i], bits[b], gen);
    }
  }
}

struct SeparateFields : ScriptGen
{
  virtual void Pre(std::ofstream& out, const char* fname, bool is_cuda) const {
    out << "src = src.SeparateFields()" << std::endl;
  }
};

struct AlignCUDA : SeparateFields
{
  virtual void Test(std::ofstream& out, const char* fname, bool is_cuda) const {
    if (is_cuda) {
      out << "srcuda = src.OnCPU(0).Align()" << std::endl;
      out << "cuda = srcuda." << fname << ".OnCUDA(0)" << std::endl;
    }
    else {
      out << "cuda = src." << fname << std::endl;
    }
  }
};

TEST_F(GenericTest, AlignCUDA)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y, FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  Test("Weave()", formats, AlignCUDA());
}

TEST_F(GenericTest, Weave)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y, FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  Test("Weave()", formats, SeparateFields());
}

TEST_F(GenericTest, DoubleWeave_Field)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y, FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  Test("DoubleWeave()", formats, SeparateFields());
}

TEST_F(GenericTest, DoubleWeave_Frame)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y, FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  Test("DoubleWeave()", formats);
}


TEST_F(GenericTest, Invert_All)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y, FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  Test("Invert()", formats);
}

TEST_F(GenericTest, Invert_R)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  Test("Invert(channels=\"R\")", formats);
}

TEST_F(GenericTest, Invert_RG)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  Test("Invert(channels=\"RG\")", formats);
}

TEST_F(GenericTest, Invert_GB)
{
  std::vector<FORMAT> formats = { FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  Test("Invert(channels=\"GB\")", formats);
}

TEST_F(GenericTest, Invert_Y)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y };
  Test("Invert(channels=\"Y\")", formats);
}

TEST_F(GenericTest, Invert_YU)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444 };
  Test("Invert(channels=\"YU\")", formats);
}

TEST_F(GenericTest, Invert_UV)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444 };
  Test("Invert(channels=\"UV\")", formats);
}

struct MergeGen : ScriptGen
{
  virtual void Pre(std::ofstream& out, const char* fname, bool is_cuda) const {
    out << "src2 = src.GaussResize(1920,1080,0,0,1920.0001,1080.0001,p=2)" << std::endl;
  }
  virtual void Test(std::ofstream& out, const char* fname, bool is_cuda) const {
    if (is_cuda) {
      out << "srcuda = src.Align().OnCPU(0)" << std::endl;
      out << "src2 = src2.Align().OnCPU(0)" << std::endl;
      out << "cuda = srcuda." << fname << ".OnCUDA(0)" << std::endl;
    }
    else {
      out << "cuda = src." << fname << std::endl;
    }
  }
};

TEST_F(GenericTest, Merge_Avg)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y, FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  Test("Merge(src2)", formats, MergeGen());
}

TEST_F(GenericTest, Merge_Merge)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y, FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  Test("Merge(src2, 0.3)", formats, MergeGen());
}

TEST_F(GenericTest, Merge_ChromaAvg)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444 };
  Test("MergeChroma(src2)", formats, MergeGen());
}

TEST_F(GenericTest, Merge_ChromaMerge)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444 };
  Test("MergeChroma(src2, 0.3)", formats, MergeGen());
}

TEST_F(GenericTest, Merge_LumaAvg)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y };
  Test("MergeLuma(src2)", formats, MergeGen());
}

TEST_F(GenericTest, Merge_LumaMerge)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y };
  Test("MergeLuma(src2, 0.3)", formats, MergeGen());
}


struct CropGen : ScriptGen
{
  virtual void Test(std::ofstream& out, const char* fname, bool is_cuda) const {
    if (is_cuda) {
      out << "srcuda = src.Align().OnCPU(0)" << std::endl;
      out << "cuda = srcuda." << fname << ".Align().OnCUDA(0)" << std::endl;
    }
    else {
      out << "cuda = src." << fname << std::endl;
    }
  }
};

TEST_F(GenericTest, Crop_2)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y, FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  Test("Crop(2,2,-2,-2)", formats, CropGen());
}

TEST_F(GenericTest, Crop_4)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y, FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  Test("Crop(4,4,-4,-4)", formats, CropGen());
}

TEST_F(GenericTest, Crop_2468)
{
  std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y, FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
  Test("Crop(2,4,-6,-8)", formats, CropGen());
}

struct ConvBitsGen : ScriptGen
{
	virtual double Thresh(
		std::ofstream& out, const char* fname, bool is_cuda, int bits) const {
		return (bits == 32) ? 1 : 0;
	}
};

struct ConvBits32Gen : ScriptGen
{
	virtual double Thresh(
		std::ofstream& out, const char* fname, bool is_cuda, int bits) const {
		return 0.25;
	}
};

TEST_F(GenericTest, ConvBitsTo8)
{
	// ビット変換はパターンとして full scale と shifted scale がある
	// full scale は 0-255 を 0-65535 にマッピングする変換
	// shifted scale は 0-255 を 0-65280 にマッピングする変換
	// CUDA版はshifted scaleしかサポートしていない
	// オプションなしだとRGBがfull scale変換されるので値が合わなくなるので注意

	std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y, FORMAT_RGB, FORMAT_RGBA, FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
	int bits[] = { 8, 16, 10, 12, 14, 32 };
	for (int i = 0; i < (int)formats.size(); ++i) {
		int nbits = IsRGB(formats[i]) ? 2 : 6;
		const char* fname = IsRGB(formats[i]) ? "ConvertBits(8,fulls=false,dither=-1)" : "ConvertBits(8,dither=-1)";
		for (int b = 0; b < nbits; ++b) {
			Test_(fname, false, formats[i], bits[b], ConvBitsGen());
			Test_(fname, true, formats[i], bits[b], ConvBitsGen());
		}
	}
}

TEST_F(GenericTest, ConvBitsTo8Dither)
{
	std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y };
	int bits[] = { 16, 10, 12, 14 };
	for (int i = 0; i < (int)formats.size(); ++i) {
		for (int b = 0; b < sizeof(bits) / sizeof(bits[0]); ++b) {
			const char* fname = "ConvertBits(8,dither=0)";
			Test_(fname, false, formats[i], bits[b], ConvBitsGen());
			Test_(fname, true, formats[i], bits[b], ConvBitsGen());
		}
	}
}

TEST_F(GenericTest, ConvBitsTo10)
{
	std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y };
	Test("ConvertBits(10,dither=-1)", formats, ConvBitsGen());
}

TEST_F(GenericTest, ConvBitsTo10Dither)
{
	std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y };
	int bits[] = { 16, 12, 14 };
	for (int i = 0; i < (int)formats.size(); ++i) {
		for (int b = 0; b < sizeof(bits) / sizeof(bits[0]); ++b) {
			const char* fname = "ConvertBits(10,dither=0)";
			Test_(fname, false, formats[i], bits[b], ConvBitsGen());
			Test_(fname, true, formats[i], bits[b], ConvBitsGen());
		}
	}
}

TEST_F(GenericTest, ConvBitsTo12)
{
	std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y };
	Test("ConvertBits(12,dither=-1)", formats, ConvBitsGen());
}

TEST_F(GenericTest, ConvBitsTo12Dither)
{
	std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y };
	int bits[] = { 16, 14 };
	for (int i = 0; i < (int)formats.size(); ++i) {
		for (int b = 0; b < sizeof(bits) / sizeof(bits[0]); ++b) {
			const char* fname = "ConvertBits(12,dither=0)";
			Test_(fname, false, formats[i], bits[b], ConvBitsGen());
			Test_(fname, true, formats[i], bits[b], ConvBitsGen());
		}
	}
}

TEST_F(GenericTest, ConvBitsTo14)
{
	std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y };
	Test("ConvertBits(14,dither=-1)", formats, ConvBitsGen());
}

TEST_F(GenericTest, ConvBitsTo14Dither)
{
	std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y };
	int bits[] = { 16 };
	for (int i = 0; i < (int)formats.size(); ++i) {
		for (int b = 0; b < sizeof(bits) / sizeof(bits[0]); ++b) {
			const char* fname = "ConvertBits(14,dither=0)";
			Test_(fname, false, formats[i], bits[b], ConvBitsGen());
			Test_(fname, true, formats[i], bits[b], ConvBitsGen());
		}
	}
}

TEST_F(GenericTest, ConvBitsTo16)
{
	std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y, FORMAT_RGB, FORMAT_RGBA, FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
	int bits[] = { 8, 16, 10, 12, 14, 32 };
	for (int i = 0; i < (int)formats.size(); ++i) {
		int nbits = IsRGB(formats[i]) ? 2 : 6;
		const char* fname = IsRGB(formats[i]) ? "ConvertBits(16,fulls=false,dither=-1)" : "ConvertBits(16,dither=-1)";
		for (int b = 0; b < nbits; ++b) {
			Test_(fname, false, formats[i], bits[b], ConvBitsGen());
			Test_(fname, true, formats[i], bits[b], ConvBitsGen());
		}
	}
}

TEST_F(GenericTest, ConvBitsTo32)
{
	std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y };
	Test("ConvertBits(32,dither=-1)", formats, ConvBits32Gen());
}

TEST_F(GenericTest, Resize_PointResize)
{
	//std::vector<FORMAT> formats = { FORMAT_YV420, FORMAT_YV422, FORMAT_YV444, FORMAT_Y, FORMAT_RGB, FORMAT_RGBA, FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
	std::vector<FORMAT> formats = { FORMAT_RGBA, FORMAT_PLANAR_RGB, FORMAT_PLANAR_RGBA };
	Test("PointResize(600,1080)", formats, ConvBits32Gen());
}
