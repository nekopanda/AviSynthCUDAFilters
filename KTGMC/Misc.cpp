#define _CRT_SECURE_NO_WARNINGS
#include "avisynth.h"

#define NOMINMAX
#include <windows.h>

#include "CommonFunctions.h"
#include "DeviceLocalData.h"
#include "DebugWriter.h"
#include "Misc.h"

#include <string>

// commonのcppを取り入れる
#include "DebugWriter.cpp"
#include "DeviceLocalData.cpp"

void AddFuncKernel(IScriptEnvironment* env);
void AddFuncMV(IScriptEnvironment* env);

static void init_console()
{
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
	freopen("CONIN$", "r", stdin);
}

void OnCudaError(cudaError_t err) {
#if 1 // デバッグ用（本番は取り除く）
	printf("[CUDA Error] %s (code: %d)\n", cudaGetErrorString(err), err);
#endif
}

int GetDeviceTypes(const PClip& clip)
{
  int devtypes = (clip->GetVersion() >= 5) ? clip->SetCacheHints(CACHE_GET_DEV_TYPE, 0) : 0;
  if (devtypes == 0) {
    return DEV_TYPE_CPU;
  }
  return devtypes;
}

class Time : public GenericVideoFilter {
	std::string name;
public:
	Time(PClip _child, const char* name, IScriptEnvironment* env)
		: GenericVideoFilter(_child)
		, name(name)
	{ }

	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
	{
		LARGE_INTEGER liBefore, liAfter, liFreq;

		QueryPerformanceCounter(&liBefore);

		PVideoFrame frame = child->GetFrame(n, env);

		QueryPerformanceCounter(&liAfter);
		QueryPerformanceFrequency(&liFreq);

		double sec = (double)(liAfter.QuadPart - liBefore.QuadPart) / liFreq.QuadPart;
		printf("[%5d] N:%5d %s: %.1f ms\n", GetCurrentThreadId(), n, name.c_str(), sec * 1000);

		return frame;
	}
};

AVSValue __cdecl Create_Time(AVSValue args, void* user_data, IScriptEnvironment* env) {
	return new Time(args[0].AsClip(), args[1].AsString("Time"), env);
}

const AVS_Linkage *AVS_linkage = 0;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors)
{
  AVS_linkage = vectors;
	//init_console();

	AddFuncKernel(env);
  AddFuncMV(env);

	return "CUDA Accelerated QTGMC Plugin";
}
