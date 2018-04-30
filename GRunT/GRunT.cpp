// GRunT - Gavino's Run-Time plugin
// =====

// An extension to the Avisynth Run-Time Environment designed to fix bugs and usability problems.


// Avisynth v2.5.  Copyright 2002 Ben Rudiak-Gould et al.
// http://www.avisynth.org

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA, or visit
// http://www.gnu.org/copyleft/gpl.html .
//
// Linking Avisynth statically or dynamically with other modules is making a
// combined work based on Avisynth.  Thus, the terms and conditions of the GNU
// General Public License cover the whole combination.
//
// As a special exception, the copyright holders of Avisynth give you
// permission to link Avisynth with independent modules that communicate with
// Avisynth solely through the interfaces defined in avisynth.h, regardless of the license
// terms of these independent modules, and to copy and distribute the
// resulting combined work under terms of your choice, provided that
// every copy of the combined work is accompanied by a complete copy of
// the source code of Avisynth (the version of Avisynth used to produce the
// combined work), being distributed under the terms of the GNU General
// Public License plus this exception.  An independent module is a module
// which is not derived from or based on Avisynth, such as 3rd-party filters,
// import and export plugins, or graphical user interfaces.


// GRunT is released under the terms of the GNU General Public License as described above.
// Copyright (C) 2008 Gavino (http://forum.doom9.org/member.php?u=141297)

// Change History:

// v1.0.1 (Gavino, 27th September 2008):
// - fix for Avisynth 2.5.7 (have to use alternative names for filters, eg GScriptClip)

// v1.0.0 (Gavino, 9th July 2008):
// - add 'args' and 'local' to filters
// - run-time functions with offset from current_frame
// - new variant of ConditionalFilter (single boolean expr)

// v0.1 (Gavino, 18th June 2008):
// - fix Avisynth bug in setting of current_frame
// - make current_frame global (allows run-time functions to be cslled inside user functions)

/* -------------------------------------------------------------------------------------- */

#include <windows.h>
#include <avisynth.h>

class Binding {
  char* name;
  AVSValue value;
  Binding* next;
public:
  Binding(char* _name, AVSValue _value) : name(_name), value(_value), next(NULL) {}
  ~Binding() { delete next; }
  Binding* Add(Binding* b) {  next = b; return b; }
  void Activate(PNeoEnv env);
  AVSValue AsString(IScriptEnvironment* env);
};

void Binding::Activate(PNeoEnv env) {
  Binding* b = this;
  do {
    env->SetGlobalVar(b->name, b->value);
    b = b->next;
  }
  while (b != NULL);
}

AVSValue Binding::AsString(IScriptEnvironment* env) { // for testing only
  Binding* b = this;
  char out[60];
  int iOut = 0;
  do {
    // add <b->name> = <b->value> <nl>
    const char *nm = b->name;
    AVSValue val = b->value;
    const char *valStr;

    strcpy(out+iOut, nm); iOut += strlen(nm);
    out[iOut++] = '=';
    if (val.IsClip()) {
      strcpy(out+iOut, "<clip>"); iOut += 6;
    }
    else if (val.IsString()) {
      out[iOut++] = '"'; out[iOut++] = '"'; out[iOut++] = '"';
      valStr = val.AsString();
      strcpy(out+iOut, valStr); iOut += strlen(valStr);
      out[iOut++] = '"'; out[iOut++] = '"'; out[iOut++] = '"';
    }
    else {
      val = env->Invoke("String", val);
      valStr = val.AsString();
      strcpy(out+iOut, valStr); iOut += strlen(valStr);
    }
    out[iOut++] = '\n';

    b = b->next;
  }
  while (b != NULL);

  out[iOut] = 0;
  return env->SaveString(out);
}

class Binder {
  const char* str;
  char c;
  int idx;
  Binding* Parse1(IScriptEnvironment* env);
  void Next() { c = str[idx++]; }
  char* SaveName(const int start, const int len, IScriptEnvironment* env);
public:
  Binder(const char* _str) : str(_str), idx(0) {}
  Binding* Parse(IScriptEnvironment* env);
};

char* Binder::SaveName(int start, int len, IScriptEnvironment* env) {
  // skip leading white space:
  while (str[start] <= 32 && len > 0) {
    start++;
    len--;
  }
  // skip trailing white space:
  while (str[start+len-1] <= 32 && len > 0) {
    len--;
  }

  // check for valid chars in name:
  for (int i=0; i<len; i++) {
    char ch = str[start+i];
    if (!(isalpha(ch) || ch == '_' || (i > 0 && isalnum(ch))))
      len = 0;
  }

  if (len == 0) env->ThrowError("args: invalid variable name");

  return env->SaveString(str+start, len);
}

Binding* Binder::Parse1(IScriptEnvironment* env) {
  char *name;
  AVSValue val;
  int start = idx, len = 0;

  Next();
  while (c != '=' && c != ',' && c != 0) {
    Next();
    len++;
  }
  name = SaveName(start, len, env);

  if (c == '=') {
    int brackets = 0;
    start = idx;
    len = 0;
    Next();
    while ((c != ',' || brackets > 0) && c != 0) {
      if (c == '(') brackets++;
      else if (c == ')') brackets--;
      Next();
      len++;
    }
  }

  if (len >= 64)
    env->ThrowError("args: expression string too long");

  char expr[64];
  strncpy(expr, str+start, len);
  expr[len] = 0;

  val = env->Invoke("Eval", expr); // errors handled by caller
  return new Binding(name, val);
}

Binding* Binder::Parse(IScriptEnvironment* env) {
// str has form: name [= expr] [, name [=expr]] ...
  Binding *first, *last;

  first = last = Parse1(env);
  while (c == ',')
    last = last->Add(Parse1(env));
  return first;
}

static AVSValue BindVars(const char* str, IScriptEnvironment* env) {
// only for testing purposes
    Binder b(str);
    Binding *binding = b.Parse(env);
    return binding->AsString(env);
}

class RTWrapper : public GenericVideoFilter {
    Binding* binding;
    bool local;
    static bool defLocal;
    Binding* BindVars(const char* str, IScriptEnvironment* env);
public:
	RTWrapper(PClip _child, AVSValue args, IScriptEnvironment* env);
	~RTWrapper() { if (binding) delete binding; }
	virtual PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
	static AVSValue Config(const bool defaults, IScriptEnvironment* env);
	static AVSValue Create(const char* name, const int argPos, const char* argName,
	                       AVSValue args, IScriptEnvironment* env);
	static AVSValue CreateConditionalFilter2(AVSValue args, IScriptEnvironment* env);
};

bool RTWrapper::defLocal = false;

AVSValue RTWrapper::Config(const bool defaults, IScriptEnvironment* env) {
    AVSValue dummy;

    defLocal = defaults;

    return dummy;
}

AVSValue RTWrapper::Create(const char* name, const int argPos, const char* argName,
                           AVSValue args, IScriptEnvironment* env) {
    const int nArgs = args.ArraySize() - 2;
    AVSValue* newArgs = new AVSValue[nArgs+1];
    const char** names = new const char*[nArgs+1];

    // build arg list for internal function:
    for (int i=0; i<nArgs; i++) {
      if (i != argPos) newArgs[i] = args[i]; // skip 'special' arg
      names[i] = NULL;
    }
    newArgs[nArgs] = args[argPos]; // special arg named at end
    names[nArgs] = argName;

    PClip inner = env->Invoke(name, AVSValue(newArgs, nArgs+1), names).AsClip();

    delete [] newArgs;
    delete [] names;

    return new RTWrapper(inner, args, env);
}

AVSValue RTWrapper::CreateConditionalFilter2(AVSValue args, IScriptEnvironment* env) {
    AVSValue newArgs[9];

    // build arg list for (extended) real function:
    for (int i=0; i<4; i++)
      newArgs[i] = args[i];
    newArgs[4] = "=";
    newArgs[5] = "true";
    for (int i=6; i<9; i++)
      newArgs[i] = args[i-2];

    return Create("ConditionalFilter", 6, "show", AVSValue(newArgs, 9), env);
}

RTWrapper::RTWrapper(PClip _child, AVSValue args, IScriptEnvironment* env_): GenericVideoFilter(_child) {
    // check NeoEnv
    PNeoEnv env = env_;
    if (!env) env_->ThrowError("GRunT: This version of GRunT only support Avisynth Neo.");

    int i = args.ArraySize() - 2; // start of 'new' args

    binding = (args[i].Defined() ? BindVars(args[i].AsString(), env) : NULL);
    local = args[i+1].AsBool(binding != NULL || defLocal);
}

Binding* RTWrapper::BindVars(const char* str, IScriptEnvironment* env) {
    Binder b(str);
    return b.Parse(env);
}

class GlobalVarFrame
{
    PNeoEnv env;
    bool enable;
public:
    GlobalVarFrame(PNeoEnv env, bool enable) : env(env), enable(enable) {
        if(enable) env->PushContextGlobal();
    }
    ~GlobalVarFrame() {
        if (enable) env->PopContextGlobal();
    }
};

PVideoFrame __stdcall RTWrapper::GetFrame(int n, IScriptEnvironment* env_) {
    PNeoEnv env = env_;
    GlobalVarFrame var_frame(env, local);
    if (binding) binding->Activate(env);
    return child->GetFrame(n, env);
}

AVSValue RTFunction(const char* name, AVSValue args, IScriptEnvironment* env) {

    AVSValue currFrame;
    AVSValue newArgs[2];
    int nArgs = 0;

    // build arg list for internal function:
    for (int i=0; i<args.ArraySize()-1; i++) {
      newArgs[nArgs++] = args[i];
    }

    try {
      currFrame = env->GetVar("current_frame");
      if (currFrame.IsInt()) {
	    env->SetVar("current_frame", currFrame.AsInt() + args[nArgs].AsInt(0));
      }
    }
    catch (IScriptEnvironment::NotFound) {}

    AVSValue result = env->Invoke(name, AVSValue(newArgs, nArgs));

    if (currFrame.Defined()) {
	  env->SetVar("current_frame", currFrame);
    }

    return result;
}

AVSValue __cdecl RTConfig(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTWrapper::Config(args[0].AsBool(true), env);
}

AVSValue __cdecl BindStr(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return BindVars(args[0].AsString(), env);
}

AVSValue __cdecl MakeScriptClip(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTWrapper::Create("ScriptClip", 2, "show", args, env);
}

AVSValue __cdecl MakeFrameEvaluate(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTWrapper::Create("FrameEvaluate", 2, "show", args, env);
}

AVSValue __cdecl MakeConditionalFilter(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTWrapper::Create("ConditionalFilter", 6, "show", args, env);
}

AVSValue __cdecl MakeConditionalFilter2(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTWrapper::CreateConditionalFilter2(args, env);
}

AVSValue __cdecl MakeWriteFile(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTWrapper::Create("WriteFile", 1, "filename", args, env);
}

AVSValue __cdecl MakeWriteFileIf(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTWrapper::Create("WriteFileIf", 1, "filename", args, env);
}

AVSValue __cdecl AverageLuma(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("AverageLuma", args, env);
}

AVSValue __cdecl AverageChromaU(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("AverageChromaU", args, env);
}

AVSValue __cdecl AverageChromaV(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("AverageChromaV", args, env);
}

AVSValue __cdecl RGBDifference(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("RGBDifference", args, env);
}

AVSValue __cdecl LumaDifference(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("LumaDifference", args, env);
}

AVSValue __cdecl ChromaUDifference(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("ChromaUDifference", args, env);
}

AVSValue __cdecl ChromaVDifference(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("ChromaVDifference", args, env);
}

AVSValue __cdecl YDifferenceFromPrevious(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("YDifferenceFromPrevious", args, env);
}

AVSValue __cdecl UDifferenceFromPrevious(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("UDifferenceFromPrevious", args, env);
}

AVSValue __cdecl VDifferenceFromPrevious(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("VDifferenceFromPrevious", args, env);
}

AVSValue __cdecl RGBDifferenceFromPrevious(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("RGBDifferenceFromPrevious", args, env);
}

AVSValue __cdecl YDifferenceToNext(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("YDifferenceToNext", args, env);
}

AVSValue __cdecl UDifferenceToNext(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("UDifferenceToNext", args, env);
}

AVSValue __cdecl VDifferenceToNext(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("VDifferenceToNext", args, env);
}

AVSValue __cdecl RGBDifferenceToNext(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("RGBDifferenceToNext", args, env);
}

AVSValue __cdecl YPlaneMax(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("YPlaneMax", args, env);
}

AVSValue __cdecl YPlaneMin(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("YPlaneMin", args, env);
}

AVSValue __cdecl YPlaneMedian(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("YPlaneMedian", args, env);
}

AVSValue __cdecl YPlaneMinMaxDifference(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("YPlaneMinMaxDifference", args, env);
}

AVSValue __cdecl UPlaneMax(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("UPlaneMax", args, env);
}

AVSValue __cdecl UPlaneMin(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("UPlaneMin", args, env);
}

AVSValue __cdecl UPlaneMedian(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("UPlaneMedian", args, env);
}

AVSValue __cdecl UPlaneMinMaxDifference(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("UPlaneMinMaxDifference", args, env);
}

AVSValue __cdecl VPlaneMax(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("VPlaneMax", args, env);
}

AVSValue __cdecl VPlaneMin(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("VPlaneMin", args, env);
}

AVSValue __cdecl VPlaneMedian(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("VPlaneMedian", args, env);
}

AVSValue __cdecl VPlaneMinMaxDifference(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return RTFunction("VPlaneMinMaxDifference", args, env);
}

const AVS_Linkage *AVS_linkage = 0;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;

    // get Avisynth version number, for compatibility check:
    //float avsVersion = 0;
    //try {
    //  AVSValue args[1];
    //  avsVersion = (float)env->Invoke("VersionNumber", AVSValue(args, 0)).AsFloat();
    //}
    //catch (IScriptEnvironment::NotFound) { env->ThrowError("GRunT: internal error"); }

    env->AddFunction("GRTConfig", "[local]b", RTConfig, 0);

    // Run-Time filters:
    env->AddFunction("GScriptClip",    "cs[show]b[after_frame]b[args]s[local]b", MakeScriptClip, 0);
    env->AddFunction("GFrameEvaluate", "cs[show]b[after_frame]b[args]s[local]b", MakeFrameEvaluate, 0);
    env->AddFunction("GConditionalFilter","cccsss[show]b[args]s[local]b", MakeConditionalFilter, 0);
    env->AddFunction("GConditionalFilter","cccs[show]b[args]s[local]b", MakeConditionalFilter2, 0);
    env->AddFunction("GWriteFile",   "c[filename]ss+[append]b[flush]b[args]s[local]b", MakeWriteFile, 0);
    env->AddFunction("GWriteFileIf", "c[filename]ss+[append]b[flush]b[args]s[local]b", MakeWriteFileIf, 0);
    //if (avsVersion >= 2.58f) {
      // mechanism for same-name filters only works on 2.58 and later
      env->AddFunction("ScriptClip",    "cs[showx]b[after_frame]b[args]s[local]b", MakeScriptClip, 0);
      env->AddFunction("FrameEvaluate", "cs[showx]b[after_frame]b[args]s[local]b", MakeFrameEvaluate, 0);
      env->AddFunction("ConditionalFilter","cccsss[showx]b[args]s[local]b", MakeConditionalFilter, 0);
      env->AddFunction("ConditionalFilter","cccs[showx]b[args]s[local]b", MakeConditionalFilter2, 0);
      env->AddFunction("WriteFile",   "c[filenamex]ss+[append]b[flush]b[args]s[local]b", MakeWriteFile, 0);
      env->AddFunction("WriteFileIf", "c[filenamex]ss+[append]b[flush]b[args]s[local]b", MakeWriteFileIf, 0);
    //}

    // Run-Time Functions:
    //env->AddFunction("AverageLuma","ci", AverageLuma, 0);
    //env->AddFunction("AverageChromaU","ci", AverageChromaU, 0);
    //env->AddFunction("AverageChromaV","ci", AverageChromaV, 0);
    //env->AddFunction("RGBDifference","cci", RGBDifference, 0);
    //env->AddFunction("LumaDifference","cci", LumaDifference, 0);
    //env->AddFunction("ChromaUDifference","cci", ChromaUDifference, 0);
    //env->AddFunction("ChromaVDifference","cci", ChromaVDifference, 0);
    //env->AddFunction("YDifferenceFromPrevious","ci", YDifferenceFromPrevious, 0);
    //env->AddFunction("UDifferenceFromPrevious","ci", UDifferenceFromPrevious, 0);
    //env->AddFunction("VDifferenceFromPrevious","ci", VDifferenceFromPrevious, 0);
    //env->AddFunction("RGBDifferenceFromPrevious","ci", RGBDifferenceFromPrevious, 0);
    //env->AddFunction("YDifferenceToNext","ci", YDifferenceToNext, 0);
    //env->AddFunction("UDifferenceToNext","ci", UDifferenceToNext, 0);
    //env->AddFunction("VDifferenceToNext","ci", VDifferenceToNext, 0);
    //env->AddFunction("RGBDifferenceToNext","ci", RGBDifferenceToNext, 0);
    //env->AddFunction("YPlaneMax","c[threshold]fi", YPlaneMax, 0);
    //env->AddFunction("YPlaneMin","c[threshold]fi", YPlaneMin, 0);
    //env->AddFunction("YPlaneMedian","ci", YPlaneMedian, 0);
    //env->AddFunction("YPlaneMinMaxDifference","c[threshold]fi", YPlaneMinMaxDifference, 0);
    //env->AddFunction("UPlaneMax","c[threshold]fi", UPlaneMax, 0);
    //env->AddFunction("UPlaneMin","c[threshold]fi", UPlaneMin, 0);
    //env->AddFunction("UPlaneMedian","ci", UPlaneMedian, 0);
    //env->AddFunction("UPlaneMinMaxDifference","c[threshold]fi", UPlaneMinMaxDifference, 0);
    //env->AddFunction("VPlaneMax","c[threshold]fi", VPlaneMax, 0);
    //env->AddFunction("VPlaneMin","c[threshold]fi", VPlaneMin, 0);
    //env->AddFunction("VPlaneMedian","ci", VPlaneMedian, 0);
    //env->AddFunction("VPlaneMinMaxDifference","c[threshold]fi", VPlaneMinMaxDifference, 0);
    // for testing:
    //env->AddFunction("BindStr","s", BindStr, 0);

    return "'GRunT' Gavino's Run-Time plugin";
}
