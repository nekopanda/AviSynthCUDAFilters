
#include <avisynth.h>
#include <stdlib.h>

#include "AvsCUDA.h"

class StaticFrame : public GenericVideoFilter {
   int n;
   PVideoFrame frame;

public:
   StaticFrame(PClip child, int n)
      : GenericVideoFilter(child)
      , n(n)
   {}
   PVideoFrame __stdcall GetFrame(int, IScriptEnvironment* env) {
      if (!frame) {
         frame = child->GetFrame(n, env);
      }
      return frame;
   }
   int __stdcall SetCacheHints(int cachehints, int frame_range)
   {
      switch (cachehints)
      {
      case CACHE_DONT_CACHE_ME:
         return 1;
      case CACHE_GET_MTMODE:
         return MT_SERIALIZED;
      case CACHE_GET_DEV_TYPE:
         return (child->GetVersion() >= 5) ? child->SetCacheHints(CACHE_GET_DEV_TYPE, 0) : 0;
      default:
         return 0;
      }
   };

   static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
   {
      return new StaticFrame(args[0].AsClip(), args[1].AsInt(0));
   }
};

extern const FuncDefinition support_filters[] = {
   { "StaticFrame",    "c[n]i", StaticFrame::Create, 0 },
   { 0 }
};
