#include "TestCommons.h"

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

void AvsTestBase::GetFrames(PClip& clip, TEST_FRAMES tf, PNeoEnv env)
{
   int nframes = clip->GetVideoInfo().num_frames;
   switch (tf) {
   case TF_MID:
      for (int i = 0; i < 8; ++i) {
         clip->GetFrame(100 + i, env);
      }
      break;
   case TF_BEGIN:
      clip->GetFrame(0, env);
      clip->GetFrame(1, env);
      clip->GetFrame(2, env);
      clip->GetFrame(3, env);
      clip->GetFrame(4, env);
      clip->GetFrame(5, env);
      break;
   case TF_END:
      clip->GetFrame(nframes - 6, env);
      clip->GetFrame(nframes - 5, env);
      clip->GetFrame(nframes - 4, env);
      clip->GetFrame(nframes - 3, env);
      clip->GetFrame(nframes - 2, env);
      clip->GetFrame(nframes - 1, env);
      break;
   }
}
