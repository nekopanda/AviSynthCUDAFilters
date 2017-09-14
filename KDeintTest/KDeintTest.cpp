
#define _CRT_SECURE_NO_WARNINGS

#include <Windows.h>

#define AVS_LINKAGE_DLLIMPORT
#include "avisynth.h"
#pragma comment(lib, "avisynth.lib")

#include "gtest/gtest.h"

#include <fstream>
#include <string>
#include <iostream>

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

// �e�X�g�ΏۂƂȂ�N���X Foo �̂��߂̃t�B�N�X�`��
class TestBase : public ::testing::Test {
protected:
	TestBase() { }

	virtual ~TestBase() {
		// �e�X�g���Ɏ��s�����C��O�𓊂��Ȃ� clean-up �������ɏ����܂��D
	}

	// �R���X�g���N�^�ƃf�X�g���N�^�ł͕s�\���ȏꍇ�D
	// �ȉ��̃��\�b�h���`���邱�Ƃ��ł��܂��F

	virtual void SetUp() {
		// ���̃R�[�h�́C�R���X�g���N�^�̒���i�e�e�X�g�̒��O�j
		// �ɌĂяo����܂��D
		char buf[MAX_PATH];
		GetModuleFileName(nullptr, buf, MAX_PATH);
		modulePath = GetDirectoryName(buf);
		workDirPath = GetDirectoryName(GetDirectoryName(modulePath)) + "\\TestScripts";
	}

	virtual void TearDown() {
		// ���̃R�[�h�́C�e�e�X�g�̒���i�f�X�g���N�^�̒��O�j
		// �ɌĂяo����܂��D
	}

	std::string modulePath;
	std::string workDirPath;

	void AnalyzeCUDATest(bool cuda, int blksize, bool chroma, int pel, int batch);
};

void TestBase::AnalyzeCUDATest(bool cuda, int blksize, bool chroma, int pel, int batch)
{
	try {
		IScriptEnvironment2* env = CreateScriptEnvironment2();

		AVSValue result;
		std::string kdeintPath = modulePath + "\\KDeint.dll";
		env->LoadPlugin(kdeintPath.c_str(), true, &result);

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
      // �o�b�`���͑S�Ċm�F����
      //for (int i = 0; i < batch; ++i) {
      for (int i = 0; i < 1; ++i) {
        clip->GetFrame(100 + i, env);
      }
		}

		env->DeleteScriptEnvironment();
	}
	catch (const AvisynthError& err) {
		printf("%s\n", err.msg);
		GTEST_FAIL();
	}
}

TEST_F(TestBase, AnalyzeCUDA_Blk32NoCPel1Batch1)
{
  AnalyzeCUDATest(true, 32, false, 1, 1);
}

TEST_F(TestBase, AnalyzeCUDA_Blk32NoCPel1Batch2)
{
  AnalyzeCUDATest(true, 32, false, 1, 2);
}

TEST_F(TestBase, AnalyzeCUDA_Blk32NoCPel1Batch3)
{
  AnalyzeCUDATest(true, 32, false, 1, 3);
}

TEST_F(TestBase, AnalyzeCUDA_Blk32NoCPel1Batch8)
{
  AnalyzeCUDATest(true, 32, false, 1, 8);
}

TEST_F(TestBase, AnalyzeCUDA_Blk16WithCPel2)
{
	AnalyzeCUDATest(true, 16, true, 2, 4);
}

TEST_F(TestBase, AnalyzeCUDA_Blk16WithCPel1)
{
	AnalyzeCUDATest(true, 16, true, 1, 4);
}

TEST_F(TestBase, AnalyzeCUDA_Blk16NoCPel2)
{
	AnalyzeCUDATest(true, 16, false, 2, 4);
}

TEST_F(TestBase, AnalyzeCUDA_Blk16NoCPel1)
{
	AnalyzeCUDATest(true, 16, false, 1, 4);
}

TEST_F(TestBase, AnalyzeCUDA_Blk32WithCPel2)
{
	AnalyzeCUDATest(true, 32, true, 2, 4);
}

TEST_F(TestBase, AnalyzeCUDA_Blk32WithCPel1)
{
	AnalyzeCUDATest(true, 32, true, 1, 4);
}

TEST_F(TestBase, AnalyzeCUDA_Blk32NoCPel2)
{
	AnalyzeCUDATest(true, 32, false, 2, 4);
}

TEST_F(TestBase, AnalyzeCUDA_Blk32NoCPel1)
{
	AnalyzeCUDATest(true, 32, false, 1, 4);
}

TEST_F(TestBase, AnalyzeCPU_Blk16WithCPel2)
{
  AnalyzeCUDATest(false, 16, true, 2, 4);
}

int main(int argc, char **argv)
{
	::testing::GTEST_FLAG(filter) = "*Analyze*";
	::testing::InitGoogleTest(&argc, argv);
	int result = RUN_ALL_TESTS();

	//getchar();

	return result;
}
