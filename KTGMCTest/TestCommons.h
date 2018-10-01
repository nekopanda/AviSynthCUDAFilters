#pragma once

#define _CRT_SECURE_NO_WARNINGS

#define AVS_LINKAGE_DLLIMPORT
#include "avisynth.h"

#define NOMINMAX
#include <Windows.h>

#include "gtest/gtest.h"

#include <fstream>
#include <string>
#include <iostream>
#include <memory>

#define O_C(n) ".OnCUDA(" #n ", 0)"

std::string GetDirectoryName(const std::string& filename);

struct ScriptEnvironmentDeleter {
  void operator()(IScriptEnvironment* env) {
    env->DeleteScriptEnvironment();
  }
};

typedef std::unique_ptr<IScriptEnvironment2, ScriptEnvironmentDeleter> PEnv;

class AvsTestBase : public ::testing::Test {
protected:
  AvsTestBase() { }

  virtual ~AvsTestBase() {
    // テスト毎に実行される，例外を投げない clean-up をここに書きます．
  }

  // コンストラクタとデストラクタでは不十分な場合．
  // 以下のメソッドを定義することができます：

  virtual void SetUp() {
    // このコードは，コンストラクタの直後（各テストの直前）
    // に呼び出されます．
    char buf[MAX_PATH];
    GetModuleFileName(nullptr, buf, MAX_PATH);
    modulePath = GetDirectoryName(buf);
    workDirPath = GetDirectoryName(GetDirectoryName(modulePath)) + "\\TestScripts";
  }

  virtual void TearDown() {
    // このコードは，各テストの直後（デストラクタの直前）
    // に呼び出されます．
  }

  std::string modulePath;
  std::string workDirPath;

  enum TEST_FRAMES {
    TF_MID, TF_BEGIN, TF_END, TF_100
  };

  void GetFrames(PClip& clip, TEST_FRAMES tf, PNeoEnv env);
};

