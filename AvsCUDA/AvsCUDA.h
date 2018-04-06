#pragma once

#include <avisynth.h>

struct FuncDefinition {
   const char* name;
   const char* params;
   IScriptEnvironment::ApplyFunc func;
   void* user_data;
};

int GetDeviceType(const PClip& clip);
