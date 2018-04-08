#include "avisynth.h"

#define NOMINMAX
#include <windows.h>
#include <Psapi.h>
#include <iostream>
#include <string>
#include <algorithm>


bool hasEnding(std::string const &fullString, std::string const &ending) {
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
  }
  else {
    return false;
  }
}

AVSValue __cdecl IsProcess(AVSValue args, void* user_data, IScriptEnvironment* env) {
	std::string exe = args[0].AsString("");
	if (exe.empty()) {
		env->ThrowError("IsProcess: exe is empty!");
	}

  char buf[MAX_PATH];
  if (GetProcessImageFileName(GetCurrentProcess(), buf, MAX_PATH)) {
    std::string name(buf);
		std::transform(name.begin(), name.end(), name.begin(), ::tolower);
		std::transform(exe.begin(), exe.end(), exe.begin(), ::tolower);
    return hasEnding(name, exe);
  }
  return false;
}

const AVS_Linkage *AVS_linkage = 0;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
  AVS_linkage = vectors;
  env->AddFunction("IsProcess", "s", IsProcess, 0);
  return "IsProcess?";
}
