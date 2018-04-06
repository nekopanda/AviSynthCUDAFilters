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

#include "focus.h"
#include <cmath>
#include <vector>
#include <avs/alignment.h>
#include "../core/internal.h"
#include <avs/minmax.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <stdint.h>
#include "../AvsCUDA.h"


/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/

extern const FuncDefinition Focus_filters[] = {
  { "Blur",           BUILTIN_FUNC_PREFIX, "cf[]f[mmx]b", Create_Blur },                     // amount [-1.0 - 1.5849625] -- log2(3)
  { "Sharpen",        BUILTIN_FUNC_PREFIX, "cf[]f[mmx]b", Create_Sharpen },               // amount [-1.5849625 - 1.0]
  { "TemporalSoften", BUILTIN_FUNC_PREFIX, "ciii[scenechange]i[mode]i", TemporalSoften::Create }, // radius, luma_threshold, chroma_threshold
  { "SpatialSoften",  BUILTIN_FUNC_PREFIX, "ciii", SpatialSoften::Create },   // radius, luma_threshold, chroma_threshold
  { 0 }
};





/****************************************
 ***  AdjustFocus helper classes     ***
 ***  Originally by Ben R.G.         ***
 ***  MMX code by Marc FD            ***
 ***  Adaptation and bugfixes sh0dan ***
 ***  Code actually requires ISSE!   ***
 ***  Not anymore - pure MMX    IanB ***
 ***  Implement boundary proc.  IanB ***
 ***  Impl. full 8bit MMX proc. IanB ***
 ***************************************/

AdjustFocusV::AdjustFocusV(double _amount, PClip _child)
: GenericVideoFilter(_child), amountd(pow(2.0, _amount)) {
    half_amount = int(32768 * amountd + 0.5);
}

template<typename pixel_t>
static void af_vertical_c(BYTE* line_buf8, BYTE* dstp8, const int height, const int pitch8, const int width, const int half_amount, int bits_per_pixel) {
  typedef typename std::conditional < sizeof(pixel_t) == 1, int, __int64>::type weight_t;
  // kernel:[(1-1/2^_amount)/2, 1/2^_amount, (1-1/2^_amount)/2]
  weight_t center_weight = half_amount*2;    // *2: 16 bit scaled arithmetic, but the converted amount parameter scaled is only 15 bits
  weight_t outer_weight = 32768-half_amount; // (1-1/2^_amount)/2  32768 = 0.5
  int max_pixel_value = (1 << bits_per_pixel) - 1;

  pixel_t * dstp = reinterpret_cast<pixel_t *>(dstp8);
  pixel_t * line_buf = reinterpret_cast<pixel_t *>(line_buf8);
  int pitch = pitch8 / sizeof(pixel_t);

  for (int y = height-1; y>0; --y) {
    for (int x = 0; x < width; ++x) {
      pixel_t a;
      // Note: ScaledPixelClip is overloaded. With __int64 parameter and uint16_t result works for 16 bit
      if(sizeof(pixel_t) == 1)
        a = ScaledPixelClip((weight_t)(dstp[x] * center_weight + (line_buf[x] + dstp[x+pitch]) * outer_weight));
      else
        a = (pixel_t)ScaledPixelClipEx((weight_t)(dstp[x] * center_weight + (line_buf[x] + dstp[x+pitch]) * outer_weight), max_pixel_value);
      line_buf[x] = dstp[x];
      dstp[x] = a;
    }
    dstp += pitch;
  }
  for (int x = 0; x < width; ++x) { // Last row - map centre as lower
    if(sizeof(pixel_t) == 1)
      dstp[x] = ScaledPixelClip((weight_t)(dstp[x] * center_weight + (line_buf[x] + dstp[x]) * outer_weight));
    else
      dstp[x] = (pixel_t)ScaledPixelClipEx((weight_t)(dstp[x] * center_weight + (line_buf[x] + dstp[x]) * outer_weight), max_pixel_value);
  }
}

static void af_vertical_c_float(BYTE* line_buf8, BYTE* dstp8, const int height, const int pitch8, const int width, const float amount) {
    float *dstp = reinterpret_cast<float *>(dstp8);
    float *line_buf = reinterpret_cast<float *>(line_buf8);
    int pitch = pitch8 / sizeof(float);

    const float center_weight = amount;
    const float outer_weight = (1.0f - amount) / 2.0f;

    for (int y = height-1; y>0; --y) {
        for (int x = 0; x < width; ++x) {
            float a = dstp[x] * center_weight + (line_buf[x] + dstp[x+pitch]) * outer_weight;
            line_buf[x] = dstp[x];
            dstp[x] = a;
        }
        dstp += pitch;
    }
    for (int x = 0; x < width; ++x) { // Last row - map centre as lower
        dstp[x] = dstp[x] * center_weight + (line_buf[x] + dstp[x]) * outer_weight;
    }
}

static void af_vertical_sse2_float(BYTE* line_buf, BYTE* dstp, const int height, const int pitch, const int row_size, const float amount) {

  const float center_weight = amount;
  const float outer_weight = (1.0f - amount) / 2.0f;

  __m128 center_weight_simd = _mm_set1_ps(center_weight);
  __m128 outer_weight_simd = _mm_set1_ps(outer_weight);
  
  for (int y = 0; y < height - 1; ++y) {
    for (int x = 0; x < row_size; x += 16) {
      __m128 upper = _mm_load_ps(reinterpret_cast<const float*>(line_buf + x));
      __m128 center = _mm_load_ps(reinterpret_cast<const float*>(dstp + x));
      __m128 lower = _mm_load_ps(reinterpret_cast<const float*>(dstp + pitch + x));
      _mm_store_ps(reinterpret_cast<float*>(line_buf + x), center);

      __m128 tmp1 = _mm_mul_ps(center, center_weight_simd);
      __m128 tmp2 = _mm_mul_ps(_mm_add_ps(upper, lower), outer_weight_simd);
      __m128 result = _mm_add_ps(tmp1, tmp2);

      _mm_store_ps(reinterpret_cast<float*>(dstp + x), result);
    }
    dstp += pitch;
  }

  //last line
  for (int x = 0; x < row_size; x += 16) {
    __m128 upper = _mm_load_ps(reinterpret_cast<const float*>(line_buf + x));
    __m128 center = _mm_load_ps(reinterpret_cast<const float*>(dstp + x));

    __m128 tmp1 = _mm_mul_ps(center, center_weight_simd);
    __m128 tmp2 = _mm_mul_ps(_mm_add_ps(upper, center), outer_weight_simd); // last line: center instead of lower
    __m128 result = _mm_add_ps(tmp1, tmp2);

    _mm_store_ps(reinterpret_cast<float*>(dstp + x), result);
  }
}


static __forceinline __m128i af_blend_sse2(__m128i &upper, __m128i &center, __m128i &lower, __m128i &center_weight, __m128i &outer_weight, __m128i &round_mask) {
  __m128i outer_tmp = _mm_add_epi16(upper, lower);
  __m128i center_tmp = _mm_mullo_epi16(center, center_weight);

  outer_tmp = _mm_mullo_epi16(outer_tmp, outer_weight);

  __m128i result = _mm_adds_epi16(center_tmp, outer_tmp);
  result = _mm_adds_epi16(result, center_tmp);
  result = _mm_adds_epi16(result, round_mask);
  return _mm_srai_epi16(result, 7);
}

template<bool useSSE4>
static __forceinline __m128i af_blend_uint16_t_sse2(__m128i &upper, __m128i &center, __m128i &lower, __m128i &center_weight, __m128i &outer_weight, __m128i &round_mask) {
  __m128i outer_tmp = _mm_add_epi32(upper, lower);
  __m128i center_tmp;
  if (useSSE4) {
    center_tmp = _mm_mullo_epi32(center, center_weight);
    outer_tmp = _mm_mullo_epi32(outer_tmp, outer_weight);
  }
  else {
    center_tmp = _MM_MULLO_EPI32(center, center_weight);
    outer_tmp = _MM_MULLO_EPI32(outer_tmp, outer_weight);
  }

  __m128i result = _mm_add_epi32(center_tmp, outer_tmp);
  result = _mm_add_epi32(result, center_tmp);
  result = _mm_add_epi32(result, round_mask);
  return _mm_srai_epi32(result, 7);
}

static __forceinline __m128 af_blend_float_sse2(__m128 &upper, __m128 &center, __m128 &lower, __m128 &center_weight, __m128 &outer_weight) {
  __m128 tmp1 = _mm_mul_ps(center, center_weight);
  __m128 tmp2 = _mm_mul_ps(_mm_add_ps(upper, lower), outer_weight);
  return _mm_add_ps(tmp1, tmp2);
}


static __forceinline __m128i af_unpack_blend_sse2(__m128i &left, __m128i &center, __m128i &right, __m128i &center_weight, __m128i &outer_weight, __m128i &round_mask, __m128i &zero) {
  __m128i left_lo = _mm_unpacklo_epi8(left, zero);
  __m128i left_hi = _mm_unpackhi_epi8(left, zero);
  __m128i center_lo = _mm_unpacklo_epi8(center, zero);
  __m128i center_hi = _mm_unpackhi_epi8(center, zero);
  __m128i right_lo = _mm_unpacklo_epi8(right, zero);
  __m128i right_hi = _mm_unpackhi_epi8(right, zero);

  __m128i result_lo = af_blend_sse2(left_lo, center_lo, right_lo, center_weight, outer_weight, round_mask);
  __m128i result_hi = af_blend_sse2(left_hi, center_hi, right_hi, center_weight, outer_weight, round_mask);

  return _mm_packus_epi16(result_lo, result_hi);
}

template<bool useSSE4>
static __forceinline __m128i af_unpack_blend_uint16_t_sse2(__m128i &left, __m128i &center, __m128i &right, __m128i &center_weight, __m128i &outer_weight, __m128i &round_mask, __m128i &zero) {
  __m128i left_lo = _mm_unpacklo_epi16(left, zero);
  __m128i left_hi = _mm_unpackhi_epi16(left, zero);
  __m128i center_lo = _mm_unpacklo_epi16(center, zero);
  __m128i center_hi = _mm_unpackhi_epi16(center, zero);
  __m128i right_lo = _mm_unpacklo_epi16(right, zero);
  __m128i right_hi = _mm_unpackhi_epi16(right, zero);

  __m128i result_lo = af_blend_uint16_t_sse2<useSSE4>(left_lo, center_lo, right_lo, center_weight, outer_weight, round_mask);
  __m128i result_hi = af_blend_uint16_t_sse2<useSSE4>(left_hi, center_hi, right_hi, center_weight, outer_weight, round_mask);
  if(useSSE4)
    return _mm_packus_epi32(result_lo, result_hi);
  else
    return _MM_PACKUS_EPI32(result_lo, result_hi);
}

template<bool useSSE4>
static void af_vertical_uint16_t_sse2(BYTE* line_buf, BYTE* dstp, int height, int pitch, int row_size, int amount) {
  // amount was: half_amount (32768). Full: 65536 (2**16)
  // now it becomes 2**(16-9)=2**7 scale
  int t = (amount + 256) >> 9; // 16-9 = 7 -> shift in 
  __m128i center_weight = _mm_set1_epi32(t);
  __m128i outer_weight = _mm_set1_epi32(64 - t);
  __m128i round_mask = _mm_set1_epi32(0x40);
  __m128i zero = _mm_setzero_si128();

  for (int y = 0; y < height - 1; ++y) {
    for (int x = 0; x < row_size; x += 16) {
      __m128i upper = _mm_load_si128(reinterpret_cast<const __m128i*>(line_buf + x));
      __m128i center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp + x));
      __m128i lower = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp + pitch + x));
      _mm_store_si128(reinterpret_cast<__m128i*>(line_buf + x), center);

      __m128i upper_lo = _mm_unpacklo_epi16(upper, zero);
      __m128i upper_hi = _mm_unpackhi_epi16(upper, zero);
      __m128i center_lo = _mm_unpacklo_epi16(center, zero);
      __m128i center_hi = _mm_unpackhi_epi16(center, zero);
      __m128i lower_lo = _mm_unpacklo_epi16(lower, zero);
      __m128i lower_hi = _mm_unpackhi_epi16(lower, zero);

      __m128i result_lo = af_blend_uint16_t_sse2<useSSE4>(upper_lo, center_lo, lower_lo, center_weight, outer_weight, round_mask);
      __m128i result_hi = af_blend_uint16_t_sse2<useSSE4>(upper_hi, center_hi, lower_hi, center_weight, outer_weight, round_mask);

      __m128i result;
      if(useSSE4)
        result = _mm_packus_epi32(result_lo, result_hi);
      else
        result = _MM_PACKUS_EPI32(result_lo, result_hi);

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), result);
    }
    dstp += pitch;
  }

  //last line
  for (int x = 0; x < row_size; x += 16) {
    __m128i upper = _mm_load_si128(reinterpret_cast<const __m128i*>(line_buf + x));
    __m128i center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp + x));

    __m128i upper_lo = _mm_unpacklo_epi16(upper, zero);
    __m128i upper_hi = _mm_unpackhi_epi16(upper, zero);
    __m128i center_lo = _mm_unpacklo_epi16(center, zero);
    __m128i center_hi = _mm_unpackhi_epi16(center, zero);

    __m128i result_lo = af_blend_uint16_t_sse2<useSSE4>(upper_lo, center_lo, center_lo, center_weight, outer_weight, round_mask);
    __m128i result_hi = af_blend_uint16_t_sse2<useSSE4>(upper_hi, center_hi, center_hi, center_weight, outer_weight, round_mask);

    __m128i result;
    if (useSSE4)
      result = _mm_packus_epi32(result_lo, result_hi);
    else
      result = _MM_PACKUS_EPI32(result_lo, result_hi);

    _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), result);
  }
}

static void af_vertical_sse2(BYTE* line_buf, BYTE* dstp, int height, int pitch, int width, int amount) {
  short t = (amount + 256) >> 9;
  __m128i center_weight = _mm_set1_epi16(t);
  __m128i outer_weight = _mm_set1_epi16(64 - t);
  __m128i round_mask = _mm_set1_epi16(0x40);
  __m128i zero = _mm_setzero_si128();

  for (int y = 0; y < height-1; ++y) {
    for (int x = 0; x < width; x+= 16) {
      __m128i upper = _mm_load_si128(reinterpret_cast<const __m128i*>(line_buf+x));
      __m128i center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp+x));
      __m128i lower = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp+pitch+x));
      _mm_store_si128(reinterpret_cast<__m128i*>(line_buf+x), center);

      __m128i upper_lo = _mm_unpacklo_epi8(upper, zero);
      __m128i upper_hi = _mm_unpackhi_epi8(upper, zero);
      __m128i center_lo = _mm_unpacklo_epi8(center, zero);
      __m128i center_hi = _mm_unpackhi_epi8(center, zero);
      __m128i lower_lo = _mm_unpacklo_epi8(lower, zero);
      __m128i lower_hi = _mm_unpackhi_epi8(lower, zero);

      __m128i result_lo = af_blend_sse2(upper_lo, center_lo, lower_lo, center_weight, outer_weight, round_mask);
      __m128i result_hi = af_blend_sse2(upper_hi, center_hi, lower_hi, center_weight, outer_weight, round_mask);

      __m128i result = _mm_packus_epi16(result_lo, result_hi);

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp+x), result);
    }
    dstp += pitch;
  }

  //last line
  for (int x = 0; x < width; x+= 16) {
    __m128i upper = _mm_load_si128(reinterpret_cast<const __m128i*>(line_buf+x));
    __m128i center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp+x));

    __m128i upper_lo = _mm_unpacklo_epi8(upper, zero);
    __m128i upper_hi = _mm_unpackhi_epi8(upper, zero);
    __m128i center_lo = _mm_unpacklo_epi8(center, zero);
    __m128i center_hi = _mm_unpackhi_epi8(center, zero);

    __m128i result_lo = af_blend_sse2(upper_lo, center_lo, center_lo, center_weight, outer_weight, round_mask);
    __m128i result_hi = af_blend_sse2(upper_hi, center_hi, center_hi, center_weight, outer_weight, round_mask);

    __m128i result = _mm_packus_epi16(result_lo, result_hi);

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp+x), result);
  }
}

#ifdef X86_32

static __forceinline __m64 af_blend_mmx(__m64 &upper, __m64 &center, __m64 &lower, __m64 &center_weight, __m64 &outer_weight, __m64 &round_mask) {
  __m64 outer_tmp = _mm_add_pi16(upper, lower);
  __m64 center_tmp = _mm_mullo_pi16(center, center_weight);

  outer_tmp = _mm_mullo_pi16(outer_tmp, outer_weight);

  __m64 result = _mm_adds_pi16(center_tmp, outer_tmp);
  result = _mm_adds_pi16(result, center_tmp);
  result = _mm_adds_pi16(result, round_mask);
  return _mm_srai_pi16(result, 7);
}

static __forceinline __m64 af_unpack_blend_mmx(__m64 &left, __m64 &center, __m64 &right, __m64 &center_weight, __m64 &outer_weight, __m64 &round_mask, __m64 &zero) {
  __m64 left_lo = _mm_unpacklo_pi8(left, zero);
  __m64 left_hi = _mm_unpackhi_pi8(left, zero);
  __m64 center_lo = _mm_unpacklo_pi8(center, zero);
  __m64 center_hi = _mm_unpackhi_pi8(center, zero);
  __m64 right_lo = _mm_unpacklo_pi8(right, zero);
  __m64 right_hi = _mm_unpackhi_pi8(right, zero);

  __m64 result_lo = af_blend_mmx(left_lo, center_lo, right_lo, center_weight, outer_weight, round_mask);
  __m64 result_hi = af_blend_mmx(left_hi, center_hi, right_hi, center_weight, outer_weight, round_mask);

  return _mm_packs_pu16(result_lo, result_hi);
}

static void af_vertical_mmx(BYTE* line_buf, BYTE* dstp, int height, int pitch, int width, int amount) {
  short t = (amount + 256) >> 9;
  __m64 center_weight = _mm_set1_pi16(t);
  __m64 outer_weight = _mm_set1_pi16(64 - t);
  __m64 round_mask = _mm_set1_pi16(0x40);
  __m64 zero = _mm_setzero_si64();

  for (int y = 0; y < height-1; ++y) {
    for (int x = 0; x < width; x+= 8) {
      __m64 upper = *reinterpret_cast<const __m64*>(line_buf+x);
      __m64 center = *reinterpret_cast<const __m64*>(dstp+x);
      __m64 lower = *reinterpret_cast<const __m64*>(dstp+pitch+x);
      *reinterpret_cast<__m64*>(line_buf+x) = center;

      __m64 result = af_unpack_blend_mmx(upper, center, lower, center_weight, outer_weight, round_mask, zero);

      *reinterpret_cast<__m64*>(dstp+x) = result;
    }
    dstp += pitch;
  }

  //last line
  for (int x = 0; x < width; x+= 8) {
    __m64 upper = *reinterpret_cast<const __m64*>(line_buf+x);
    __m64 center = *reinterpret_cast<const __m64*>(dstp+x);

    __m64 upper_lo = _mm_unpacklo_pi8(upper, zero);
    __m64 upper_hi = _mm_unpackhi_pi8(upper, zero);
    __m64 center_lo = _mm_unpacklo_pi8(center, zero);
    __m64 center_hi = _mm_unpackhi_pi8(center, zero);

    __m64 result_lo = af_blend_mmx(upper_lo, center_lo, center_lo, center_weight, outer_weight, round_mask);
    __m64 result_hi = af_blend_mmx(upper_hi, center_hi, center_hi, center_weight, outer_weight, round_mask);

    __m64 result = _mm_packs_pu16(result_lo, result_hi);

    *reinterpret_cast<__m64*>(dstp+x) = result;
  }
  _mm_empty();
}

#endif

template<typename pixel_t>
static void af_vertical_process(BYTE* line_buf, BYTE* dstp, size_t height, size_t pitch, size_t row_size, int half_amount, int bits_per_pixel, IScriptEnvironment* env) {
  size_t width = row_size / sizeof(pixel_t);
  // only for 8/16 bit, float separated
  if (sizeof(pixel_t) == 1 && (env->GetCPUFlags() & CPUF_AVX2) && IsPtrAligned(dstp, 32) && width >= 32) {
    //pitch of aligned frames is always >= 32 so we'll just process some garbage if width is not mod32
    af_vertical_avx2(line_buf, dstp, (int)height, (int)pitch, (int)width, half_amount);
  }
  else
  if (sizeof(pixel_t) == 1 && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(dstp, 16) && width >= 16) {
    //pitch of aligned frames is always >= 16 so we'll just process some garbage if width is not mod16
    af_vertical_sse2(line_buf, dstp, (int)height, (int)pitch, (int)width, half_amount);
  }
  else if (sizeof(pixel_t) == 2 && (env->GetCPUFlags() & CPUF_AVX2) && IsPtrAligned(dstp, 32) && row_size >= 32) {
    af_vertical_uint16_t_avx2(line_buf, dstp, (int)height, (int)pitch, (int)row_size, half_amount);
  }
  else if (sizeof(pixel_t) == 2 && (env->GetCPUFlags() & CPUF_SSE4_1) && IsPtrAligned(dstp, 16) && row_size >= 16) {
    af_vertical_uint16_t_sse2<true>(line_buf, dstp, (int)height, (int)pitch, (int)row_size, half_amount);
  }
  else if (sizeof(pixel_t) == 2 && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(dstp, 16) && row_size >= 16) {
    af_vertical_uint16_t_sse2<false>(line_buf, dstp, (int)height, (int)pitch, (int)row_size, half_amount);
  }
  else
#ifdef X86_32
  if (sizeof(pixel_t) == 1 && (env->GetCPUFlags() & CPUF_MMX) && width >= 8)
  {
    size_t mod8_width = width / 8 * 8;
    af_vertical_mmx(line_buf, dstp, height, pitch, mod8_width, half_amount);
    if (mod8_width != width) {
      //yes, this is bad for caching. MMX shouldn't be used these days anyway
      af_vertical_c<uint8_t>(line_buf, dstp + mod8_width, height, pitch, width - mod8_width, half_amount, bits_per_pixel);
    }
  } else
#endif
  {
      af_vertical_c<pixel_t>(line_buf, dstp, (int)height, (int)pitch, (int)width, half_amount, bits_per_pixel);
  }
}

static void af_vertical_process_float(BYTE* line_buf, BYTE* dstp, size_t height, size_t pitch, size_t row_size, double amountd, IScriptEnvironment* env) {
    size_t width = row_size / sizeof(float);
    if ((env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(dstp, 16) && width >= 16) {
        //pitch of aligned frames is always >= 16 so we'll just process some garbage if width is not mod16
        af_vertical_sse2_float(line_buf, dstp, (int)height, (int)pitch, (int)row_size, (float)amountd);
    } else {
        af_vertical_c_float(line_buf, dstp, (int)height, (int)pitch, (int)width, (float)amountd);
    }
}

// --------------------------------
// Vertical Blur/Sharpen
// --------------------------------

PVideoFrame __stdcall AdjustFocusV::GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame src = child->GetFrame(n, env);

    env->MakeWritable(&src);

    auto env2 = static_cast<IScriptEnvironment2*>(env);
    BYTE* line_buf = reinterpret_cast<BYTE*>(env2->Allocate(AlignNumber(src->GetRowSize(), FRAME_ALIGN), FRAME_ALIGN, AVS_POOLED_ALLOC));
    if (!line_buf) {
        env2->ThrowError("AdjustFocusV: Could not reserve memory.");
    }

    int pixelsize = vi.ComponentSize();
    int bits_per_pixel = vi.BitsPerComponent();

    if (vi.IsPlanar()) {
      const int planesYUV[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A};
      const int planesRGB[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A};
      const int *planes = vi.IsYUV() || vi.IsYUVA() ? planesYUV : planesRGB;

      for (int cplane = 0; cplane < 3; cplane++) {
            int plane = planes[cplane];
            BYTE* dstp = src->GetWritePtr(plane);
            int pitch = src->GetPitch(plane);
            int row_size = src->GetRowSize(plane);
            int height = src->GetHeight(plane);
            memcpy(line_buf, dstp, row_size); // First row - map centre as upper

            switch (pixelsize) {
            case 1: af_vertical_process<uint8_t>(line_buf, dstp, height, pitch, row_size, half_amount, bits_per_pixel, env); break;
            case 2: af_vertical_process<uint16_t>(line_buf, dstp, height, pitch, row_size, half_amount, bits_per_pixel, env); break;
            default: // 4: float
                af_vertical_process_float(line_buf, dstp, height, pitch, row_size, amountd, env); break;
            }
        }
    }
    else {
        BYTE* dstp = src->GetWritePtr();
        int pitch = src->GetPitch();
        int row_size = vi.RowSize();
        int height = vi.height;
        memcpy(line_buf, dstp, row_size); // First row - map centre as upper
        if (pixelsize == 1)
          af_vertical_process<uint8_t>(line_buf, dstp, height, pitch, row_size, half_amount, bits_per_pixel, env);
        else
          af_vertical_process<uint16_t>(line_buf, dstp, height, pitch, row_size, half_amount, bits_per_pixel, env);
    }

    env2->Free(line_buf);
    return src;
}


AdjustFocusH::AdjustFocusH(double _amount, PClip _child)
: GenericVideoFilter(_child), amountd(pow(2.0, _amount)) {
    half_amount = int(32768 * amountd + 0.5);
}


// --------------------------------------
// Blur/Sharpen Horizontal RGB32 C++ Code
// --------------------------------------

template<typename pixel_t, typename weight_t>
static __forceinline void af_horizontal_rgb32_process_line_c(pixel_t b_left, pixel_t g_left, pixel_t r_left, pixel_t a_left, pixel_t *dstp, size_t width, weight_t center_weight, weight_t outer_weight) {
  size_t x;
  for (x = 0; x < width-1; ++x)
  {
    pixel_t b = ScaledPixelClip((weight_t)(dstp[x*4+0] * center_weight + (b_left + dstp[x*4+4]) * outer_weight));
    b_left = dstp[x*4+0];
    dstp[x*4+0] = b;
    pixel_t g = ScaledPixelClip((weight_t)(dstp[x*4+1] * center_weight + (g_left + dstp[x*4+5]) * outer_weight));
    g_left = dstp[x*4+1];
    dstp[x*4+1] = g;
    pixel_t r = ScaledPixelClip((weight_t)(dstp[x*4+2] * center_weight + (r_left + dstp[x*4+6]) * outer_weight));
    r_left = dstp[x*4+2];
    dstp[x*4+2] = r;
    pixel_t a = ScaledPixelClip((weight_t)(dstp[x*4+3] * center_weight + (a_left + dstp[x*4+7]) * outer_weight));
    a_left = dstp[x*4+3];
    dstp[x*4+3] = a;
  }
  dstp[x*4+0] = ScaledPixelClip((weight_t)(dstp[x*4+0] * center_weight + (b_left + dstp[x*4+0]) * outer_weight));
  dstp[x*4+1] = ScaledPixelClip((weight_t)(dstp[x*4+1] * center_weight + (g_left + dstp[x*4+1]) * outer_weight));
  dstp[x*4+2] = ScaledPixelClip((weight_t)(dstp[x*4+2] * center_weight + (r_left + dstp[x*4+2]) * outer_weight));
  dstp[x*4+3] = ScaledPixelClip((weight_t)(dstp[x*4+3] * center_weight + (a_left + dstp[x*4+3]) * outer_weight));
}

template<typename pixel_t>
static void af_horizontal_rgb32_64_c(BYTE* dstp8, size_t height, size_t pitch8, size_t width, int half_amount) {
  typedef typename std::conditional < sizeof(pixel_t) == 1, int, __int64>::type weight_t;
  // kernel:[(1-1/2^_amount)/2, 1/2^_amount, (1-1/2^_amount)/2]
  weight_t center_weight = half_amount*2;    // *2: 16 bit scaled arithmetic, but the converted amount parameter scaled is only 15 bits
  weight_t outer_weight = 32768-half_amount; // (1-1/2^_amount)/2  32768 = 0.5

  pixel_t* dstp = reinterpret_cast<pixel_t *>(dstp8);
  size_t pitch = pitch8 / sizeof(pixel_t);

  for (size_t y = height; y>0; --y)
  {
    pixel_t b_left = dstp[0];
    pixel_t g_left = dstp[1];
    pixel_t r_left = dstp[2];
    pixel_t a_left = dstp[3];
    af_horizontal_rgb32_process_line_c<pixel_t, weight_t>(b_left, g_left, r_left, a_left, dstp, width, center_weight, outer_weight);
    dstp += pitch;
  }

}


//implementation is not in-place. Unaligned reads will be slow on older intels but who cares
static void af_horizontal_rgb32_sse2(BYTE* dstp, const BYTE* srcp, size_t dst_pitch, size_t src_pitch, size_t height, size_t width, size_t amount) {
  size_t width_bytes = width * 4;
  size_t loop_limit = width_bytes - 16;
  int center_weight_c = int(amount*2);
  int outer_weight_c = int(32768-amount);

  short t = short((amount + 256) >> 9);
  __m128i center_weight = _mm_set1_epi16(t);
  __m128i outer_weight = _mm_set1_epi16(64 - t);
  __m128i round_mask = _mm_set1_epi16(0x40);
  __m128i zero = _mm_setzero_si128();
//#pragma warning(disable: 4309)
  __m128i left_mask = _mm_set_epi32(0, 0, 0, 0xFFFFFFFF);
  __m128i right_mask = _mm_set_epi32(0xFFFFFFFF, 0, 0, 0);
//#pragma warning(default: 4309)

  __m128i center, right, left, result;

  for (size_t y = 0; y < height; ++y) {
    center = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp));
    right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + 4));
    left = _mm_or_si128(_mm_and_si128(center, left_mask), _mm_slli_si128(center, 4));

    result = af_unpack_blend_sse2(left, center, right, center_weight, outer_weight, round_mask, zero);

    _mm_store_si128(reinterpret_cast< __m128i*>(dstp), result);

    for (size_t x = 16; x < loop_limit; x+=16) {
      left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x - 4));
      center = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x));
      right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x + 4));

      result = af_unpack_blend_sse2(left, center, right, center_weight, outer_weight, round_mask, zero);

      _mm_store_si128(reinterpret_cast< __m128i*>(dstp+x), result);
    }

    left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + loop_limit - 4));
    center = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + loop_limit));
    right = _mm_or_si128(_mm_and_si128(center, right_mask), _mm_srli_si128(center, 4));

    result = af_unpack_blend_sse2(left, center, right, center_weight, outer_weight, round_mask, zero);

    _mm_storeu_si128(reinterpret_cast< __m128i*>(dstp + loop_limit), result);


    dstp += dst_pitch;
    srcp += src_pitch;
  }
}

template<bool useSSE4>
static void af_horizontal_rgb64_sse2(BYTE* dstp, const BYTE* srcp, size_t dst_pitch, size_t src_pitch, size_t height, size_t width, size_t amount) {
  // width is really width
  size_t width_bytes = width * 4 * sizeof(uint16_t);
  size_t loop_limit = width_bytes - 16;
  int center_weight_c = int(amount * 2);
  int outer_weight_c = int(32768 - amount);

  short t = short((amount + 256) >> 9);
  __m128i center_weight = _mm_set1_epi32(t);
  __m128i outer_weight = _mm_set1_epi32(64 - t);
  __m128i round_mask = _mm_set1_epi32(0x40);
  __m128i zero = _mm_setzero_si128();
  //#pragma warning(disable: 4309)
  __m128i left_mask = _mm_set_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF);
  __m128i right_mask = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0);
  //#pragma warning(default: 4309)

  __m128i center, right, left, result;

  for (size_t y = 0; y < height; ++y) {
    center = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp));
    right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + 4*sizeof(uint16_t))); // move right by one 4*uint16_t pixelblock
    left = _mm_or_si128(_mm_and_si128(center, left_mask), _mm_slli_si128(center, 8));

    result = af_unpack_blend_sse2(left, center, right, center_weight, outer_weight, round_mask, zero);

    _mm_store_si128(reinterpret_cast< __m128i*>(dstp), result);

    for (size_t x = 16; x < loop_limit; x += 16) {
      left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x - 4 * sizeof(uint16_t)));
      center = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x));
      right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x + 4 * sizeof(uint16_t)));

      result = af_unpack_blend_uint16_t_sse2<useSSE4>(left, center, right, center_weight, outer_weight, round_mask, zero);

      _mm_store_si128(reinterpret_cast< __m128i*>(dstp + x), result);
    }

    left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + loop_limit - 4 * sizeof(uint16_t)));
    center = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + loop_limit));
    right = _mm_or_si128(_mm_and_si128(center, right_mask), _mm_srli_si128(center, 4 * sizeof(uint16_t)));

    result = af_unpack_blend_uint16_t_sse2<useSSE4>(left, center, right, center_weight, outer_weight, round_mask, zero);

    _mm_storeu_si128(reinterpret_cast< __m128i*>(dstp + loop_limit), result);


    dstp += dst_pitch;
    srcp += src_pitch;
  }
}

#ifdef X86_32

static void af_horizontal_rgb32_mmx(BYTE* dstp, const BYTE* srcp, size_t dst_pitch, size_t src_pitch, size_t height, size_t width, size_t amount) {
  size_t width_bytes = width * 4;
  size_t loop_limit = width_bytes - 8;
  int center_weight_c = amount*2;
  int outer_weight_c = 32768-amount;

  short t = short((amount + 256) >> 9);
  __m64 center_weight = _mm_set1_pi16(t);
  __m64 outer_weight = _mm_set1_pi16(64 - t);
  __m64 round_mask = _mm_set1_pi16(0x40);
  __m64 zero = _mm_setzero_si64();
  //#pragma warning(disable: 4309)
  __m64 left_mask = _mm_set_pi32(0, 0xFFFFFFFF);
  __m64 right_mask = _mm_set_pi32(0xFFFFFFFF, 0);
  //#pragma warning(default: 4309)

  __m64 center, right, left, result;

  for (size_t y = 0; y < height; ++y) {
    center = *reinterpret_cast<const __m64*>(srcp);
    right = *reinterpret_cast<const __m64*>(srcp + 4);
    left = _mm_or_si64(_mm_and_si64(center, left_mask), _mm_slli_si64(center, 32));

    result = af_unpack_blend_mmx(left, center, right, center_weight, outer_weight, round_mask, zero);

    *reinterpret_cast< __m64*>(dstp) = result;

    for (size_t x = 8; x < loop_limit; x+=8) {
      left = *reinterpret_cast<const __m64*>(srcp + x - 4);
      center = *reinterpret_cast<const __m64*>(srcp + x);
      right = *reinterpret_cast<const __m64*>(srcp + x + 4);

      result = af_unpack_blend_mmx(left, center, right, center_weight, outer_weight, round_mask, zero);

      *reinterpret_cast< __m64*>(dstp+x) = result;
    }

    left = *reinterpret_cast<const __m64*>(srcp + loop_limit - 4);
    center = *reinterpret_cast<const __m64*>(srcp + loop_limit);
    right = _mm_or_si64(_mm_and_si64(center, right_mask), _mm_srli_si64(center, 32));

    result = af_unpack_blend_mmx(left, center, right, center_weight, outer_weight, round_mask, zero);

    *reinterpret_cast< __m64*>(dstp + loop_limit) = result;

    dstp += dst_pitch;
    srcp += src_pitch;
  }
  _mm_empty();
}

#endif

// -------------------------------------
// Blur/Sharpen Horizontal YUY2 C++ Code
// -------------------------------------

static void af_horizontal_yuy2_c(BYTE* p, int height, int pitch, int width, int amount) {
  const int center_weight = amount*2;
  const int outer_weight = 32768-amount;
  for (int y0 = height; y0>0; --y0)
  {
    BYTE yy = p[0];
    BYTE uv = p[1];
    BYTE vu = p[3];
    int x;
    for (x = 0; x < width-2; ++x)
    {
      BYTE y = ScaledPixelClip(p[x*2+0] * center_weight + (yy + p[x*2+2]) * outer_weight);
      yy   = p[x*2+0];
      p[x*2+0] = y;
      BYTE w = ScaledPixelClip(p[x*2+1] * center_weight + (uv + p[x*2+5]) * outer_weight);
      uv   = vu;
      vu   = p[x*2+1];
      p[x*2+1] = w;
    }
    BYTE y     = ScaledPixelClip(p[x*2+0] * center_weight + (yy + p[x*2+2]) * outer_weight);
    yy       = p[x*2+0];
    p[x*2+0] = y;
    p[x*2+1] = ScaledPixelClip(p[x*2+1] * center_weight + (uv + p[x*2+1]) * outer_weight);
    p[x*2+2] = ScaledPixelClip(p[x*2+2] * center_weight + (yy + p[x*2+2]) * outer_weight);
    p[x*2+3] = ScaledPixelClip(p[x*2+3] * center_weight + (vu + p[x*2+3]) * outer_weight);

    p += pitch;
  }
}


static __forceinline __m128i af_blend_yuy2_sse2(__m128i &left, __m128i &center, __m128i &right, __m128i &luma_mask,
                                             __m128i &center_weight, __m128i &outer_weight, __m128i &round_mask) {
  __m128i left_luma = _mm_and_si128(left, luma_mask); //0 Y5 0 Y4 0 Y3 0 Y2 0 Y1 0 Y0 0 Y-1 0 Y-2
  __m128i center_luma = _mm_and_si128(center, luma_mask); //0 Y7 0 Y6 0 Y5 0 Y4 0 Y3 0 Y2 0 Y1 0 Y0
  __m128i right_luma = _mm_and_si128(right, luma_mask); //0 Y9 0 Y8 0 Y7 0 Y6 0 Y5 0 Y4 0 Y3 0 Y2

  left_luma = _mm_or_si128(
    _mm_srli_si128(left_luma, 2), // 0  0 0 Y5 0 Y4 0 Y3 0 Y2 0 Y1 0 Y0 0 Y-1
    _mm_slli_si128(right_luma, 6) // 0 Y6 0 Y5 0 Y4 0 Y3 0 Y2 0  0 0  0 0  0
    ); // Y6..Y0 (Y-1)

  right_luma = _mm_or_si128(
    _mm_srli_si128(center_luma, 2),//0 0  0 Y7 0 Y6 0 Y5 0 Y4 0 Y3 0 Y2 0 Y1
    _mm_slli_si128(right_luma, 2)  //0 Y8 0 Y7 0 Y6 0 Y5 0 Y4 0 Y3 0 Y2 0 0 
    ); // Y8..Y1

  __m128i result_luma = af_blend_sse2(left_luma, center_luma, right_luma, center_weight, outer_weight, round_mask);

  __m128i left_chroma = _mm_srli_epi16(left, 8); //0 V 0 U 0 V 0 U
  __m128i center_chroma = _mm_srli_epi16(center, 8); //0 V 0 U 0 V 0 U
  __m128i right_chroma = _mm_srli_epi16(right, 8); //0 V 0 U 0 V 0 U

  __m128i result_chroma = af_blend_sse2(left_chroma, center_chroma, right_chroma, center_weight, outer_weight, round_mask);

  __m128i lo_lu_hi_co = _mm_packus_epi16(result_luma, result_chroma); // U3 V3 U2 V2 U1 V1 U0 V0 Y7 Y6 Y5 Y4 Y3 Y2 Y1 Y0
  __m128i result = _mm_unpacklo_epi8(lo_lu_hi_co, _mm_srli_si128(lo_lu_hi_co, 8)); // U3 Y7 V3 Y6 U2 Y5 V2 Y4 U1 Y3 V1 Y2 U0 Y1 V0 Y0
  return result;
}


static void af_horizontal_yuy2_sse2(BYTE* dstp, const BYTE* srcp, size_t dst_pitch, size_t src_pitch, size_t height, size_t width, size_t amount) {
  size_t width_bytes = width * 2;
  size_t loop_limit = width_bytes - 16;

  short t = short((amount + 256) >> 9);
  __m128i center_weight = _mm_set1_epi16(t);
  __m128i outer_weight = _mm_set1_epi16(64 - t);
  __m128i round_mask = _mm_set1_epi16(0x40);
#pragma warning(push)
#pragma warning(disable: 4309)
  __m128i left_mask = _mm_set_epi32(0, 0, 0, 0xFFFFFFFF);
  __m128i right_mask = _mm_set_epi32(0xFFFFFFFF, 0, 0, 0);
  __m128i left_mask_small = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0x00FF, 0);
  __m128i right_mask_small = _mm_set_epi16(0, 0x00FF, 0, 0, 0, 0, 0, 0);
  __m128i luma_mask = _mm_set1_epi16(0xFF);
#pragma warning(pop)

  __m128i center, right, left, result;

  for (size_t y = 0; y < height; ++y) {
    center = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp));//V1 Y3 U1 Y2 V0 Y1 U0 Y0
    right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + 4));//V2 Y5 U2 Y4 V1 Y3 U1 Y2

    //todo: now this is dumb
    left = _mm_or_si128(
      _mm_and_si128(center, left_mask),
      _mm_slli_si128(center, 4)
      );//V0 Y1 U0 Y0 V0 Y1 U0 Y0
    left = _mm_or_si128(
      _mm_andnot_si128(left_mask_small, left),
      _mm_and_si128(_mm_slli_si128(center, 2), left_mask_small)
      );//V0 Y1 U0 Y0 V0 Y0 U0 Y0

    result = af_blend_yuy2_sse2(left, center, right, luma_mask, center_weight, outer_weight, round_mask);

    _mm_store_si128(reinterpret_cast< __m128i*>(dstp), result);

    for (size_t x = 16; x < loop_limit; x+=16) {
      left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x - 4));//V0 Y1 U0 Y0 V-1 Y-1 U-1 Y-2
      center = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x)); //V1 Y3 U1 Y2 V0 Y1 U0 Y0
      right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x + 4));//V2 Y5 U2 Y4 V1 Y3 U1 Y2

      __m128i result = af_blend_yuy2_sse2(left, center, right, luma_mask, center_weight, outer_weight, round_mask);

      _mm_store_si128(reinterpret_cast< __m128i*>(dstp+x), result);
    }

    left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + loop_limit - 4));
    center = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + loop_limit));  //V1 Y3 U1 Y2 V0 Y1 U0 Y0

    //todo: now this is dumb2
    right = _mm_or_si128(
      _mm_and_si128(center, right_mask),
      _mm_srli_si128(center, 4)
      );//V1 Y3 U1 Y2 V1 Y3 U1 Y2

    right = _mm_or_si128(
      _mm_andnot_si128(right_mask_small, right),
      _mm_and_si128(_mm_srli_si128(center, 2), right_mask_small)
      );//V1 Y3 U1 Y3 V1 Y3 U1 Y2

    result = af_blend_yuy2_sse2(left, center, right, luma_mask, center_weight, outer_weight, round_mask);

    _mm_storeu_si128(reinterpret_cast< __m128i*>(dstp + loop_limit), result);

    dstp += dst_pitch;
    srcp += src_pitch;
  }
}



#ifdef X86_32
// -------------------------------------
// Blur/Sharpen Horizontal YUY2 MMX Code
// -------------------------------------
//
static __forceinline __m64 af_blend_yuy2_mmx(__m64 &left, __m64 &center, __m64 &right, __m64 &luma_mask,
                           __m64 &center_weight, __m64 &outer_weight, __m64 &round_mask) {
  __m64 left_luma = _mm_and_si64(left, luma_mask); //0 Y1 0 Y0 0 Y-1 0 Y-2
  __m64 center_luma = _mm_and_si64(center, luma_mask); //0 Y3 0 Y2 0 Y1 0 Y0
  __m64 right_luma = _mm_and_si64(right, luma_mask); //0 Y5 0 Y4 0 Y3 0 Y2

  left_luma = _mm_or_si64(
    _mm_srli_si64(left_luma, 16), // 0 0 0 Y1 0 Y0 0 Y-1
    _mm_slli_si64(right_luma, 48) // 0 Y2 0 0 0 0 0 0
    );

  right_luma = _mm_or_si64(
    _mm_srli_si64(center_luma, 16),//0 0 0 Y3 0 Y2 0 Y1
    _mm_slli_si64(right_luma, 16)//0 Y4 0 Y3 0 Y2 0 0
    );

  __m64 result_luma = af_blend_mmx(left_luma, center_luma, right_luma, center_weight, outer_weight, round_mask);

  __m64 left_chroma = _mm_srli_pi16(left, 8); //0 V 0 U 0 V 0 U
  __m64 center_chroma = _mm_srli_pi16(center, 8); //0 V 0 U 0 V 0 U
  __m64 right_chroma = _mm_srli_pi16(right, 8); //0 V 0 U 0 V 0 U

  __m64 result_chroma = af_blend_mmx(left_chroma, center_chroma, right_chroma, center_weight, outer_weight, round_mask);
  
  __m64 lo_lu_hi_co = _m_packuswb(result_luma, result_chroma); // U1 V1 U0 V0 Y3 Y2 Y1 Y0
  __m64 result = _mm_unpacklo_pi8(lo_lu_hi_co, _mm_srli_si64(lo_lu_hi_co, 32)); // U1 Y3 V1 Y2 U0 Y1 V0 Y0
  return result;
}


static void af_horizontal_yuy2_mmx(BYTE* dstp, const BYTE* srcp, size_t dst_pitch, size_t src_pitch, size_t height, size_t width, size_t amount) {
  size_t width_bytes = width * 2;
  size_t loop_limit = width_bytes - 8;

  short t = short((amount + 256) >> 9);
  __m64 center_weight = _mm_set1_pi16(t);
  __m64 outer_weight = _mm_set1_pi16(64 - t);
  __m64 round_mask = _mm_set1_pi16(0x40);
#pragma warning(push)
#pragma warning(disable: 4309)
  __m64 left_mask = _mm_set_pi32(0, 0xFFFFFFFF);
  __m64 right_mask = _mm_set_pi32(0xFFFFFFFF, 0);
  __m64 left_mask_small = _mm_set_pi16(0, 0, 0x00FF, 0);
  __m64 right_mask_small = _mm_set_pi16(0, 0x00FF, 0, 0);
  __m64 luma_mask = _mm_set1_pi16(0xFF);
#pragma warning(pop)

  __m64 center, right, left, result;

  for (size_t y = 0; y < height; ++y) {
    center = *reinterpret_cast<const __m64*>(srcp);//V1 Y3 U1 Y2 V0 Y1 U0 Y0
    right = *reinterpret_cast<const __m64*>(srcp + 4);//V2 Y5 U2 Y4 V1 Y3 U1 Y2

    //todo: now this is dumb
    left = _mm_or_si64(
      _mm_and_si64(center, left_mask),
      _mm_slli_si64(center, 32)
      );//V0 Y1 U0 Y0 V0 Y1 U0 Y0
    left = _mm_or_si64(
      _mm_andnot_si64(left_mask_small, left),
      _mm_and_si64(_mm_slli_si64(center, 16), left_mask_small)
      );//V0 Y1 U0 Y0 V0 Y0 U0 Y0

    result = af_blend_yuy2_mmx(left, center, right, luma_mask, center_weight, outer_weight, round_mask);

    *reinterpret_cast< __m64*>(dstp) = result;

    for (size_t x = 8; x < loop_limit; x+=8) {
      left = *reinterpret_cast<const __m64*>(srcp + x - 4);//V0 Y1 U0 Y0 V-1 Y-1 U-1 Y-2
      center = *reinterpret_cast<const __m64*>(srcp + x); //V1 Y3 U1 Y2 V0 Y1 U0 Y0
      right = *reinterpret_cast<const __m64*>(srcp + x + 4);//V2 Y5 U2 Y4 V1 Y3 U1 Y2

      __m64 result = af_blend_yuy2_mmx(left, center, right, luma_mask, center_weight, outer_weight, round_mask);

      *reinterpret_cast< __m64*>(dstp+x) = result;
    }

    left = *reinterpret_cast<const __m64*>(srcp + loop_limit - 4);
    center = *reinterpret_cast<const __m64*>(srcp + loop_limit);  //V1 Y3 U1 Y2 V0 Y1 U0 Y0

    //todo: now this is dumb2
    right = _mm_or_si64(
      _mm_and_si64(center, right_mask),
      _mm_srli_si64(center, 32)
      );//V1 Y3 U1 Y2 V1 Y3 U1 Y2
    right = _mm_or_si64(
      _mm_andnot_si64(right_mask_small, right),
      _mm_and_si64(_mm_srli_si64(center, 16), right_mask_small)
      );//V1 Y3 U1 Y3 V1 Y3 U1 Y2

    result = af_blend_yuy2_mmx(left, center, right, luma_mask, center_weight, outer_weight, round_mask);

    *reinterpret_cast< __m64*>(dstp + loop_limit) = result;

    dstp += dst_pitch;
    srcp += src_pitch;
  }
  _mm_empty();
}


#endif

// --------------------------------------
// Blur/Sharpen Horizontal RGB24 C++ Code
// --------------------------------------

template<typename pixel_t>
static void af_horizontal_rgb24_48_c(BYTE* dstp8, int height, int pitch8, int width, int half_amount) {
  typedef typename std::conditional < sizeof(pixel_t) == 1, int, __int64>::type weight_t;
  // kernel:[(1-1/2^_amount)/2, 1/2^_amount, (1-1/2^_amount)/2]
  weight_t center_weight = half_amount*2;    // *2: 16 bit scaled arithmetic, but the converted amount parameter scaled is only 15 bits
  weight_t outer_weight = 32768-half_amount; // (1-1/2^_amount)/2  32768 = 0.5

  pixel_t *dstp = reinterpret_cast<pixel_t *>(dstp8);
  int pitch = pitch8 / sizeof(pixel_t);
  for (int y = height; y>0; --y)
  {
    pixel_t bb = dstp[0];
    pixel_t gg = dstp[1];
    pixel_t rr = dstp[2];
    int x;
    for (x = 0; x < width-1; ++x)
    {
      // ScaledPixelClip has 2 overloads: BYTE/uint16_t (int/int64 i)
      pixel_t b = ScaledPixelClip((weight_t)(dstp[x*3+0] * center_weight + (bb + dstp[x*3+3]) * outer_weight));
      bb = dstp[x*3+0]; dstp[x*3+0] = b;
      pixel_t g = ScaledPixelClip((weight_t)(dstp[x*3+1] * center_weight + (gg + dstp[x*3+4]) * outer_weight));
      gg = dstp[x*3+1]; dstp[x*3+1] = g;
      pixel_t r = ScaledPixelClip((weight_t)(dstp[x*3+2] * center_weight + (rr + dstp[x*3+5]) * outer_weight));
      rr = dstp[x*3+2]; dstp[x*3+2] = r;
    }
    dstp[x*3+0] = ScaledPixelClip((weight_t)(dstp[x*3+0] * center_weight + (bb + dstp[x*3+0]) * outer_weight));
    dstp[x*3+1] = ScaledPixelClip((weight_t)(dstp[x*3+1] * center_weight + (gg + dstp[x*3+1]) * outer_weight));
    dstp[x*3+2] = ScaledPixelClip((weight_t)(dstp[x*3+2] * center_weight + (rr + dstp[x*3+2]) * outer_weight));
    dstp += pitch;
  }
}

// -------------------------------------
// Blur/Sharpen Horizontal YV12 C++ Code
// -------------------------------------

template<typename pixel_t>
static __forceinline void af_horizontal_planar_process_line_c(pixel_t left, BYTE *dstp8, size_t row_size, int center_weight, int outer_weight) {
  size_t x;
  pixel_t* dstp = reinterpret_cast<pixel_t *>(dstp8);
  typedef typename std::conditional < sizeof(pixel_t) == 1, int, __int64>::type weight_t; // for calling the right ScaledPixelClip()
  size_t width = row_size / sizeof(pixel_t);
  for (x = 0; x < width-1; ++x) {
    pixel_t temp = ScaledPixelClip((weight_t)(dstp[x] * (weight_t)center_weight + (left + dstp[x+1]) * (weight_t)outer_weight));
    left = dstp[x];
    dstp[x] = temp;
  }
  // ScaledPixelClip has 2 overloads: BYTE/uint16_t (int/int64 i)
  dstp[x] = ScaledPixelClip((weight_t)(dstp[x] * (weight_t)center_weight + (left + dstp[x]) * (weight_t)outer_weight));
}

static __forceinline void af_horizontal_planar_process_line_uint16_c(uint16_t left, BYTE *dstp8, size_t row_size, int center_weight, int outer_weight, int bits_per_pixel) {
  size_t x;
  typedef uint16_t pixel_t;
  pixel_t* dstp = reinterpret_cast<pixel_t *>(dstp8);
  const int max_pixel_value = (1 << bits_per_pixel) - 1; // clamping on 10-12-14-16 bitdepth
  typedef std::conditional < sizeof(pixel_t) == 1, int, __int64>::type weight_t; // for calling the right ScaledPixelClip()
  size_t width = row_size / sizeof(pixel_t);
  for (x = 0; x < width-1; ++x) {
    pixel_t temp = (pixel_t)ScaledPixelClipEx((weight_t)(dstp[x] * (weight_t)center_weight + (left + dstp[x+1]) * (weight_t)outer_weight), max_pixel_value);
    left = dstp[x];
    dstp[x] = temp;
  }
  // ScaledPixelClip has 2 overloads: BYTE/uint16_t (int/int64 i)
  dstp[x] = ScaledPixelClipEx((weight_t)(dstp[x] * (weight_t)center_weight + (left + dstp[x]) * (weight_t)outer_weight), max_pixel_value);
}

template<typename pixel_t>
static void af_horizontal_planar_c(BYTE* dstp8, size_t height, size_t pitch8, size_t row_size, size_t half_amount, int bits_per_pixel)
{
    pixel_t* dstp = reinterpret_cast<pixel_t *>(dstp8);
    size_t pitch = pitch8 / sizeof(pixel_t);
    int center_weight = int(half_amount*2);
    int outer_weight = int(32768-half_amount);
    pixel_t left;
    for (size_t y = height; y>0; --y) {
        left = dstp[0];
        if(sizeof(pixel_t) == 1)
          af_horizontal_planar_process_line_c<pixel_t>(left, (BYTE *)dstp, row_size, center_weight, outer_weight);
        else
          af_horizontal_planar_process_line_uint16_c(left, (BYTE *)dstp, row_size, center_weight, outer_weight, bits_per_pixel);
        dstp += pitch;
    }
}

static __forceinline void af_horizontal_planar_process_line_float_c(float left, float *dstp, size_t row_size, float center_weight, float outer_weight) {
    size_t x;
    size_t width = row_size / sizeof(float);
    for (x = 0; x < width-1; ++x) {
        float temp = dstp[x] * center_weight + (left + dstp[x+1]) * outer_weight;
        left = dstp[x];
        dstp[x] = temp;
    }
    dstp[x] = dstp[x] * center_weight + (left + dstp[x]) * outer_weight;
}

static void af_horizontal_planar_float_c(BYTE* dstp8, size_t height, size_t pitch8, size_t row_size, float amount)
{
    float* dstp = reinterpret_cast<float *>(dstp8);
    size_t pitch = pitch8 / sizeof(float);
    float center_weight = amount;
    float outer_weight = (1.0f - amount) / 2.0f;
    float left;
    for (size_t y = height; y>0; --y) {
        left = dstp[0];
        af_horizontal_planar_process_line_float_c(left, dstp, row_size, center_weight, outer_weight);
        dstp += pitch;
    }
}

static void af_horizontal_planar_sse2(BYTE* dstp, size_t height, size_t pitch, size_t width, size_t amount) {
  size_t mod16_width = (width / 16) * 16;
  size_t sse_loop_limit = width == mod16_width ? mod16_width - 16 : mod16_width;
  int center_weight_c = int(amount*2);
  int outer_weight_c = int(32768-amount);

  short t = short((amount + 256) >> 9);
  __m128i center_weight = _mm_set1_epi16(t);
  __m128i outer_weight = _mm_set1_epi16(64 - t);
  __m128i round_mask = _mm_set1_epi16(0x40);
  __m128i zero = _mm_setzero_si128();
  __m128i left_mask = _mm_set_epi32(0, 0, 0, 0xFF);
#pragma warning(push)
#pragma warning(disable: 4309)
  __m128i right_mask = _mm_set_epi8(0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
#pragma warning(pop)

  __m128i left;

  for (size_t y = 0; y < height; ++y) {
    //left border
    __m128i center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp));
    __m128i right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp+1));
    left = _mm_or_si128(_mm_and_si128(center, left_mask),  _mm_slli_si128(center, 1));

    __m128i result = af_unpack_blend_sse2(left, center, right, center_weight, outer_weight, round_mask, zero);
    left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp+15));
    _mm_store_si128(reinterpret_cast<__m128i*>(dstp), result);

    //main processing loop
    for (size_t x = 16; x < sse_loop_limit; x+= 16) {
      center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp+x));
      right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp+x+1));

      result = af_unpack_blend_sse2(left, center, right, center_weight, outer_weight, round_mask, zero);

      left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp+x+15));

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp+x), result);
    }

    //right border
    if(mod16_width == width) { //width is mod8, process with mmx
      center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp+mod16_width-16));
      right = _mm_or_si128(_mm_and_si128(center, right_mask),  _mm_srli_si128(center, 1));

      result = af_unpack_blend_sse2(left, center, right, center_weight, outer_weight, round_mask, zero);

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp+mod16_width-16), result);
    } else { //some stuff left
      BYTE l = _mm_cvtsi128_si32(left) & 0xFF;
      af_horizontal_planar_process_line_c<uint8_t>(l, dstp+mod16_width, width-mod16_width, center_weight_c, outer_weight_c);

    }

    dstp += pitch;
  }
}

template<bool useSSE4>
static void af_horizontal_planar_uint16_t_sse2(BYTE* dstp, size_t height, size_t pitch, size_t row_size, size_t amount, int bits_per_pixel) {
  size_t mod16_width = (row_size / 16) * 16;
  size_t sse_loop_limit = row_size == mod16_width ? mod16_width - 16 : mod16_width;
  int center_weight_c = int(amount * 2);
  int outer_weight_c = int(32768 - amount);

  int t = int((amount + 256) >> 9);
  __m128i center_weight = _mm_set1_epi32(t);
  __m128i outer_weight = _mm_set1_epi32(64 - t);
  __m128i round_mask = _mm_set1_epi32(0x40);
  __m128i zero = _mm_setzero_si128();
#pragma warning(push)
#pragma warning(disable: 4309)
  __m128i left_mask = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0xFFFF); // 0, 0, 0, 0, 0, 0, 0, FFFF
  __m128i right_mask = _mm_set_epi16(0xFFFF, 0, 0, 0, 0, 0, 0, 0);
#pragma warning(pop)

  __m128i left;

  for (size_t y = 0; y < height; ++y) {
    //left border
    __m128i center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp));
    __m128i right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp + 2));
    left = _mm_or_si128(_mm_and_si128(center, left_mask), _mm_slli_si128(center, 2));

    __m128i result = af_unpack_blend_uint16_t_sse2<useSSE4>(left, center, right, center_weight, outer_weight, round_mask, zero);
    left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp + (16 - 2)));
    _mm_store_si128(reinterpret_cast<__m128i*>(dstp), result);

    //main processing loop
    for (size_t x = 16; x < sse_loop_limit; x += 16) {
      center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp + x));
      right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp + x + 2));

      result = af_unpack_blend_uint16_t_sse2<useSSE4>(left, center, right, center_weight, outer_weight, round_mask, zero);

      left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp + x + (16 - 2)));

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), result);
    }

    //right border
    if (mod16_width == row_size) { //width is mod8, process with mmx
      center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp + mod16_width - 16));
      right = _mm_or_si128(_mm_and_si128(center, right_mask), _mm_srli_si128(center, 2));

      result = af_unpack_blend_uint16_t_sse2<useSSE4>(left, center, right, center_weight, outer_weight, round_mask, zero);

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + mod16_width - 16), result);
    }
    else { //some stuff left
      uint16_t l = _mm_cvtsi128_si32(left) & 0xFFFF;
      af_horizontal_planar_process_line_uint16_c(l, dstp + mod16_width, row_size - mod16_width, center_weight_c, outer_weight_c, bits_per_pixel);
    }

    dstp += pitch;
  }
}

static void af_horizontal_planar_float_sse2(BYTE* dstp, size_t height, size_t pitch, size_t row_size, float amount) {
  const float center_weight = amount;
  const float outer_weight = (1.0f - amount) / 2.0f;

  __m128 center_weight_simd = _mm_set1_ps(center_weight);
  __m128 outer_weight_simd = _mm_set1_ps(outer_weight);

  size_t mod16_width = (row_size / 16) * 16;
  size_t sse_loop_limit = row_size == mod16_width ? mod16_width - 16 : mod16_width;

#pragma warning(push)
#pragma warning(disable: 4309)
  __m128i left_mask = _mm_set_epi32(0, 0, 0, 0xFFFFFFFF);
  __m128i right_mask = _mm_set_epi32(0xFFFFFFFF, 0, 0, 0);
#pragma warning(pop)

  __m128 left;

  for (size_t y = 0; y < height; ++y) {
    //left border
    __m128 center = _mm_load_ps(reinterpret_cast<const float*>(dstp));
    __m128 right = _mm_loadu_ps(reinterpret_cast<const float*>(dstp + sizeof(float)));
    left = _mm_castsi128_ps(_mm_or_si128(_mm_and_si128(_mm_castps_si128(center), left_mask), _mm_slli_si128(_mm_castps_si128(center), sizeof(float))));

    __m128 result = af_blend_float_sse2(left, center, right, center_weight_simd, outer_weight_simd);
    left = _mm_loadu_ps(reinterpret_cast<const float*>(dstp + (16 - sizeof(float))));
    _mm_store_ps(reinterpret_cast<float*>(dstp), result);

    //main processing loop
    for (size_t x = 16; x < sse_loop_limit; x += 16) {
      center = _mm_load_ps(reinterpret_cast<const float*>(dstp + x));
      right = _mm_loadu_ps(reinterpret_cast<const float*>(dstp + x + sizeof(float)));

      result = af_blend_float_sse2(left, center, right, center_weight_simd, outer_weight_simd);

      left = _mm_loadu_ps(reinterpret_cast<const float*>(dstp + x + (16 - sizeof(float))));

      _mm_store_ps(reinterpret_cast<float*>(dstp + x), result);
    }

    //right border
    if (mod16_width == row_size) { //width is mod8, process with mmx
      center = _mm_load_ps(reinterpret_cast<const float*>(dstp + mod16_width - 16));
      right = _mm_castsi128_ps(_mm_or_si128(_mm_and_si128(_mm_castps_si128(center), right_mask), _mm_srli_si128(_mm_castps_si128(center), sizeof(float))));

      result = af_blend_float_sse2(left, center, right, center_weight_simd, outer_weight_simd);

      _mm_store_ps(reinterpret_cast<float*>(dstp + mod16_width - 16), result);
    }
    else { //some stuff left
      float l = _mm_cvtss_f32(left);
      af_horizontal_planar_process_line_float_c(l, (float *)(dstp + mod16_width), row_size - mod16_width, center_weight, outer_weight);
    }

    dstp += pitch;
  }
}


#ifdef X86_32

static void af_horizontal_planar_mmx(BYTE* dstp, size_t height, size_t pitch, size_t width, size_t amount) {
  size_t mod8_width = (width / 8) * 8;
  size_t mmx_loop_limit = width == mod8_width ? mod8_width - 8 : mod8_width;
  int center_weight_c = amount*2;
  int outer_weight_c = 32768-amount;

  short t = short((amount + 256) >> 9);
  __m64 center_weight = _mm_set1_pi16(t);
  __m64 outer_weight = _mm_set1_pi16(64 - t);
  __m64 round_mask = _mm_set1_pi16(0x40);
  __m64 zero = _mm_setzero_si64();
#pragma warning(push)
#pragma warning(disable: 4309)
  __m64 left_mask = _mm_set_pi8(0, 0, 0, 0, 0, 0, 0, 0xFF);
  __m64 right_mask = _mm_set_pi8(0xFF, 0, 0, 0, 0, 0, 0, 0);
#pragma warning(pop)

  __m64 left;

  for (size_t y = 0; y < height; ++y) {
    //left border
    __m64 center = *reinterpret_cast<const __m64*>(dstp);
    __m64 right = *reinterpret_cast<const __m64*>(dstp+1);
    left = _mm_or_si64(_mm_and_si64(center, left_mask),  _mm_slli_si64(center, 8));

    __m64 result = af_unpack_blend_mmx(left, center, right, center_weight, outer_weight, round_mask, zero);
    left = *reinterpret_cast<const __m64*>(dstp+7);
    *reinterpret_cast<__m64*>(dstp) = result;

    //main processing loop
    for (size_t x = 8; x < mmx_loop_limit; x+= 8) {
      center = *reinterpret_cast<const __m64*>(dstp+x);
      right = *reinterpret_cast<const __m64*>(dstp+x+1);

      result = af_unpack_blend_mmx(left, center, right, center_weight, outer_weight, round_mask, zero);
      left = *reinterpret_cast<const __m64*>(dstp+x+7);

      *reinterpret_cast<__m64*>(dstp+x) = result;
    }

    //right border
    if(mod8_width == width) { //width is mod8, process with mmx
      center = *reinterpret_cast<const __m64*>(dstp+mod8_width-8);
      right = _mm_or_si64(_mm_and_si64(center, right_mask),  _mm_srli_si64(center, 8));

      result = af_unpack_blend_mmx(left, center, right, center_weight, outer_weight, round_mask, zero);

      *reinterpret_cast<__m64*>(dstp+mod8_width-8) = result;
    } else { //some stuff left
      BYTE l = _mm_cvtsi64_si32(left) & 0xFF;
      af_horizontal_planar_process_line_c<uint8_t>(l, dstp+mod8_width, width-mod8_width, center_weight_c, outer_weight_c);
    }

    dstp += pitch;
  }
  _mm_empty();
}


#endif


static void copy_frame(const PVideoFrame &src, PVideoFrame &dst, IScriptEnvironment *env, const int *planes, int plane_count) {
  for (int p = 0; p < plane_count; p++) {
    int plane = planes[p];
    env->BitBlt(dst->GetWritePtr(plane), dst->GetPitch(plane), src->GetReadPtr(plane),
      src->GetPitch(plane), src->GetRowSize(plane), src->GetHeight(plane));
  }
}

// ----------------------------------
// Blur/Sharpen Horizontal GetFrame()
// ----------------------------------

PVideoFrame __stdcall AdjustFocusH::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrame(vi);

  const int planesYUV[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A};
  const int planesRGB[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A};
  const int *planes = vi.IsYUV() || vi.IsYUVA() ? planesYUV : planesRGB;

  int pixelsize = vi.ComponentSize();

  if (vi.IsPlanar()) {
    copy_frame(src, dst, env, planes, vi.NumComponents() ); //planar processing is always in-place
    int pixelsize = vi.ComponentSize();
    int bits_per_pixel = vi.BitsPerComponent();
    for(int cplane=0;cplane<3;cplane++) {
      int plane = planes[cplane];
      int row_size = dst->GetRowSize(plane);
      BYTE* q = dst->GetWritePtr(plane);
      int pitch = dst->GetPitch(plane);
      int height = dst->GetHeight(plane);
      if (pixelsize == 1 && (env->GetCPUFlags() & CPUF_AVX2) && IsPtrAligned(q, 32) && row_size > 32) {
        af_horizontal_planar_avx2(q, height, pitch, row_size, half_amount);
      }
      else
        if (pixelsize==1 && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(q, 16) && row_size > 16) {
        af_horizontal_planar_sse2(q, height, pitch, row_size, half_amount);
      } else
#ifdef X86_32
        if (pixelsize == 1 && (env->GetCPUFlags() & CPUF_MMX) && row_size > 8) {
          af_horizontal_planar_mmx(q,height,pitch,row_size,half_amount);
        } else
#endif
        if (pixelsize == 2 && (env->GetCPUFlags() & CPUF_AVX2) && IsPtrAligned(q, 32) && row_size > 32) {
          af_horizontal_planar_uint16_t_avx2(q, height, pitch, row_size, half_amount, bits_per_pixel);
        } 
        else if (pixelsize == 2 && (env->GetCPUFlags() & CPUF_SSE4_1) && IsPtrAligned(q, 16) && row_size > 16) {
          af_horizontal_planar_uint16_t_sse2<true>(q, height, pitch, row_size, half_amount, bits_per_pixel);
        } 
        else if (pixelsize == 2 && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(q, 16) && row_size > 16) {
          af_horizontal_planar_uint16_t_sse2<false>(q, height, pitch, row_size, half_amount, bits_per_pixel);
        }
        else if (pixelsize == 4 && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(q, 16) && row_size > 16) {
          af_horizontal_planar_float_sse2(q, height, pitch, row_size, (float)amountd);
        }
        else {
          switch (pixelsize) {
          case 1: af_horizontal_planar_c<uint8_t>(q, height, pitch, row_size, half_amount, bits_per_pixel); break;
          case 2: af_horizontal_planar_c<uint16_t>(q, height, pitch, row_size, half_amount, bits_per_pixel); break;
          default: // 4: float
            af_horizontal_planar_float_c(q, height, pitch, row_size, (float)amountd); break;
          }

        }
    }
  } else {
    if (vi.IsYUY2()) {
      BYTE* q = dst->GetWritePtr();
      const int pitch = dst->GetPitch();
      if ((env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(src->GetReadPtr(), 16) && vi.width>8) {
        af_horizontal_yuy2_sse2(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), vi.height, vi.width, half_amount);
      } else
#ifdef X86_32
      if ((env->GetCPUFlags() & CPUF_MMX) && vi.width>8) {
        af_horizontal_yuy2_mmx(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), vi.height, vi.width, half_amount);
      } else
#endif
      {
        copy_frame(src, dst, env, planesYUV, 1); //in-place
        af_horizontal_yuy2_c(q,vi.height,pitch,vi.width,half_amount);
      }
    }
    else if (vi.IsRGB32() || vi.IsRGB64()) {
      if ((pixelsize==1) && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(src->GetReadPtr(), 16) && vi.width>4) {
        //this one is NOT in-place
        af_horizontal_rgb32_sse2(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), vi.height, vi.width, half_amount);
      }
      else if ((pixelsize == 2) && (env->GetCPUFlags() & CPUF_SSE4_1) && IsPtrAligned(src->GetReadPtr(), 16) && vi.width > 2) {
        //this one is NOT in-place
        af_horizontal_rgb64_sse2<true>(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), vi.height, vi.width, half_amount); // really width
      }
      else if ((pixelsize == 2) && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(src->GetReadPtr(), 16) && vi.width > 2) {
        //this one is NOT in-place
        af_horizontal_rgb64_sse2<false>(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), vi.height, vi.width, half_amount); // really width
      }
      else
#ifdef X86_32
      if ((pixelsize==1) && (env->GetCPUFlags() & CPUF_MMX) && vi.width > 2)
      { //so as this one
        af_horizontal_rgb32_mmx(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), vi.height, vi.width, half_amount);
      } else
#endif
      {
        copy_frame(src, dst, env, planesYUV, 1);
        if(pixelsize==1)
          af_horizontal_rgb32_64_c<uint8_t>(dst->GetWritePtr(), vi.height, dst->GetPitch(), vi.width, half_amount);
        else
          af_horizontal_rgb32_64_c<uint16_t>(dst->GetWritePtr(), vi.height, dst->GetPitch(), vi.width, half_amount);
      }
    } else if (vi.IsRGB24() || vi.IsRGB48()) {
      copy_frame(src, dst, env, planesYUV, 1);
      if(pixelsize==1)
        af_horizontal_rgb24_48_c<uint8_t>(dst->GetWritePtr(), vi.height, dst->GetPitch(), vi.width, half_amount);
      else
        af_horizontal_rgb24_48_c<uint16_t>(dst->GetWritePtr(), vi.height, dst->GetPitch(), vi.width, half_amount);
    }
  }

  return dst;
}


/************************************************
 *******   Sharpen/Blur Factory Methods   *******
 ***********************************************/

AVSValue __cdecl Create_Sharpen(AVSValue args, void*, IScriptEnvironment* env)
{
  const double amountH = args[1].AsFloat(), amountV = args[2].AsDblDef(amountH);

  if (amountH < -1.5849625 || amountH > 1.0 || amountV < -1.5849625 || amountV > 1.0) // log2(3)
    env->ThrowError("Sharpen: arguments must be in the range -1.58 to 1.0");

  if (fabs(amountH) < 0.00002201361136) { // log2(1+1/65536)
    if (fabs(amountV) < 0.00002201361136) {
      return args[0].AsClip();
    }
    else {
      return new AdjustFocusV(amountV, args[0].AsClip());
    }
  }
  else {
    if (fabs(amountV) < 0.00002201361136) {
      return new AdjustFocusH(amountH, args[0].AsClip());
    }
    else {
      return new AdjustFocusH(amountH, new AdjustFocusV(amountV, args[0].AsClip()));
    }
  }
}

AVSValue __cdecl Create_Blur(AVSValue args, void*, IScriptEnvironment* env)
{
  const double amountH = args[1].AsFloat(), amountV = args[2].AsDblDef(amountH);
  const bool mmx = args[3].AsBool(true) && (env->GetCPUFlags() & CPUF_MMX);

  if (amountH < -1.0 || amountH > 1.5849625 || amountV < -1.0 || amountV > 1.5849625) // log2(3)
    env->ThrowError("Blur: arguments must be in the range -1.0 to 1.58");

  if (fabs(amountH) < 0.00002201361136) { // log2(1+1/65536)
    if (fabs(amountV) < 0.00002201361136) {
      return args[0].AsClip();
    }
    else {
      return new AdjustFocusV(-amountV, args[0].AsClip());
    }
  }
  else {
    if (fabs(amountV) < 0.00002201361136) {
      return new AdjustFocusH(-amountH, args[0].AsClip());
    }
    else {
      return new AdjustFocusH(-amountH, new AdjustFocusV(-amountV, args[0].AsClip()));
    }
  }
}




/***************************
 ****  TemporalSoften  *****
 **************************/

TemporalSoften::TemporalSoften( PClip _child, unsigned radius, unsigned luma_thresh,
                                unsigned chroma_thresh, int _scenechange, IScriptEnvironment* env )
  : GenericVideoFilter  (_child),
    chroma_threshold    (min(chroma_thresh,255u)),
    luma_threshold      (min(luma_thresh,255u)),
    kernel              (2*min(radius,(unsigned int)MAX_RADIUS)+1),
    scenechange (_scenechange)
{

  child->SetCacheHints(CACHE_WINDOW,kernel);

  if (vi.IsRGB24() || vi.IsRGB48()) {
    env->ThrowError("TemporalSoften: RGB24/48 Not supported, use ConvertToRGB32/48().");
  }

  if ((vi.IsRGB32() || vi.IsRGB64()) && (vi.width&1)) {
    env->ThrowError("TemporalSoften: RGB32/64 source must be multiple of 2 in width.");
  }

  if ((vi.IsYUY2()) && (vi.width&3)) {
    env->ThrowError("TemporalSoften: YUY2 source must be multiple of 4 in width.");
  }

  if (scenechange >= 255) {
    scenechange = 0;
  }

  if (scenechange>0 && (vi.IsRGB32() || vi.IsRGB64())) {
      env->ThrowError("TemporalSoften: Scenechange not available on RGB32/64");
  }

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();

  // original scenechange parameter always 0-255
  int factor;
  if (vi.IsPlanar()) // Y/YUV, no Planar RGB here
    factor = 1; // bitdepth independent. sad normalizes
  else
    factor = vi.BytesFromPixels(1) / pixelsize; // /pixelsize: correction for packed 16 bit rgb
  scenechange *= ((vi.width/32)*32)*vi.height*factor; // why /*32?


  int c = 0;
  if (vi.IsPlanar() && (vi.IsYUV() || vi.IsYUVA())) {
    if (luma_thresh>0) {
      planes[c].planeId = PLANAR_Y;
      planes[c++].threshold = luma_thresh;
    }
    if (chroma_thresh>0) {
      planes[c].planeId = PLANAR_V;
      planes[c++].threshold =chroma_thresh;
      planes[c].planeId = PLANAR_U;
      planes[c++].threshold = chroma_thresh;
    }
  } else if (vi.IsYUY2()) {
    planes[c].planeId=0;
    planes[c++].threshold=luma_thresh|(chroma_thresh<<8);
  } else if (vi.IsRGB()) {  // For RGB We use Luma.
    if (vi.IsPlanar()) {
      planes[c].planeId = PLANAR_G;
      planes[c++].threshold = luma_thresh;
      planes[c].planeId = PLANAR_B;
      planes[c++].threshold = luma_thresh;
      planes[c].planeId = PLANAR_R;
      planes[c++].threshold = luma_thresh;
    }
    else { // packed RGB
      planes[c].planeId = 0;
      planes[c++].threshold = luma_thresh;
    }
  }
  planes[c].planeId=0;
}

//offset is the initial value of x. Used when C routine processes only parts of frames after SSE/MMX paths do their job.
template<typename pixel_t, bool maxThreshold>
static void accumulate_line_c(BYTE* _c_plane, const BYTE** planeP, int planes, int offset, size_t rowsize, BYTE _threshold, int div, int bits_per_pixel) {
  pixel_t *c_plane = reinterpret_cast<pixel_t *>(_c_plane);

  typedef typename std::conditional < sizeof(pixel_t) <= 2, int, float>::type sum_t;
  typedef typename std::conditional < sizeof(pixel_t) == 1, int, int64_t>::type bigsum_t;
  typedef typename std::conditional < std::is_floating_point<pixel_t>::value, float, int>::type threshold_t;

  size_t width = rowsize / sizeof(pixel_t);

  threshold_t threshold = _threshold; // parameter 0..255
  if (std::is_floating_point<pixel_t>::value)
    threshold = (threshold_t)(threshold / 255.0f); // float
  else if (sizeof(pixel_t) == 2) {
    threshold = threshold * (uint16_t)(1 << (bits_per_pixel - 8)); // uint16_t, 10 bit: *4 16bit: *256
  }

  for (size_t x = offset; x < width; ++x) {
    pixel_t current = c_plane[x];
    sum_t sum = current;

    for (int plane = planes - 1; plane >= 0; plane--) {
      pixel_t p = reinterpret_cast<const pixel_t *>(planeP[plane])[x];
      if (maxThreshold) {
        sum += p; // simple frame average mode
      }
      else {
        sum_t absdiff = std::abs(current - p);

        if (absdiff <= threshold) {
          sum += p;
        }
        else {
          sum += current;
        }
      }
    }
    if (std::is_floating_point<pixel_t>::value)
      c_plane[x] = (pixel_t)(sum / (planes + 1)); // float: simple average
    else
      c_plane[x] = (pixel_t)(((bigsum_t)sum * div + 16384) >> 15); // div = 32768/(planes+1) for integer arithmetic
  }
}

static void accumulate_line_yuy2_c(BYTE* c_plane, const BYTE** planeP, int planes, size_t width, BYTE threshold_luma, BYTE threshold_chroma, int div) {
  for (size_t x = 0; x < width; x+=2) {
    BYTE current_y = c_plane[x];
    BYTE current_c = c_plane[x+1];
    size_t sum_y = current_y;
    size_t sum_c = current_c;

    for (int plane = planes - 1; plane >= 0; plane--) {
      BYTE p_y = planeP[plane][x];
      BYTE p_c = planeP[plane][x+1];
      size_t absdiff_y = std::abs(current_y - p_y);
      size_t absdiff_c = std::abs(current_c - p_c);

      if (absdiff_y <= threshold_luma) {
        sum_y += p_y;
      } else {
        sum_y += current_y;
      }

      if (absdiff_c <= threshold_chroma) {
        sum_c += p_c;
      } else {
        sum_c += current_c;
      }
    }

    c_plane[x] = (BYTE)((sum_y * div + 16384) >> 15);
    c_plane[x+1] = (BYTE)((sum_c * div + 16384) >> 15);
  }
}

static __forceinline __m128i ts_multiply_repack_sse2(const __m128i &src, const __m128i &div, __m128i &halfdiv, __m128i &zero) {
  __m128i acc = _mm_madd_epi16(src, div);
  acc = _mm_add_epi32(acc, halfdiv);
  acc = _mm_srli_epi32(acc, 15);
  acc = _mm_packs_epi32(acc, acc);
  return _mm_packus_epi16(acc, zero);
}

static inline __m128i _mm_cmple_epu8(__m128i x, __m128i y)
{
  // Returns 0xFF where x <= y:
  return _mm_cmpeq_epi8(_mm_min_epu8(x, y), x);
}

template<bool hasSSE4>
static inline __m128i _mm_cmple_epu16(__m128i x, __m128i y)
{
  // Returns 0xFFFF where x <= y:
  if(hasSSE4)
    return _mm_cmpeq_epi16(_mm_min_epu16(x, y), x);
  else
    return _mm_cmpeq_epi16(_mm_subs_epu16(x, y), _mm_setzero_si128());
}

// fast: maxThreshold (255) simple accumulate for average
template<bool maxThreshold, bool hasSSSE3>
static void accumulate_line_sse2(BYTE* c_plane, const BYTE** planeP, int planes, size_t width, int threshold, int div) {

  // threshold: 8 bits: 2 bytes in a word. for YUY2: luma<<8 | chroma
  // 16 bits: 16 bit value (orig threshold scaled by bits_per_pixel)
  __m128i halfdiv_vector, div_vector;
  if (hasSSSE3) {
    halfdiv_vector = _mm_set1_epi32(16384); // for mulhrs
    div_vector = _mm_set1_epi16(div);
  }
  else {
    halfdiv_vector = _mm_set1_epi16(1); // High16(0x10000)
    div_vector = _mm_set1_epi16(65536 / (planes + 1)); // mulhi
  }
  __m128i thresh = _mm_set1_epi16(threshold);

  for (size_t x = 0; x < width; x+=16) {
    __m128i current = _mm_load_si128(reinterpret_cast<const __m128i*>(c_plane+x));
    __m128i zero = _mm_setzero_si128();
     __m128i low = _mm_unpacklo_epi8(current, zero);
     __m128i high = _mm_unpackhi_epi8(current, zero);

    for (int plane = planes - 1; plane >= 0; --plane) {
      __m128i p = _mm_load_si128(reinterpret_cast<const __m128i*>(planeP[plane] + x));

      __m128i add_low, add_high;
      if (maxThreshold) {
        // fast: simple accumulate for average
        add_low = _mm_unpacklo_epi8(p, zero);
        add_high = _mm_unpackhi_epi8(p, zero);
      } else {
        auto pc = _mm_subs_epu8(p, current); // r2507-
        auto cp = _mm_subs_epu8(current, p);
        auto abs_cp = _mm_or_si128(pc, cp);
        auto leq_thresh = _mm_cmple_epu8(abs_cp, thresh);
        /*
        __m128i p_greater_t = _mm_subs_epu8(p, thresh);
        __m128i c_greater_t = _mm_subs_epu8(current, thresh);
        __m128i over_thresh = _mm_or_si128(p_greater_t, c_greater_t); //abs(p-c) - t == (satsub(p,c) | satsub(c,p)) - t =kinda= satsub(p,t) | satsub(c,t)

        __m128i leq_thresh = _mm_cmpeq_epi8(over_thresh, zero); //abs diff lower or equal to threshold
        */
        __m128i andop = _mm_and_si128(leq_thresh, p);
        __m128i andnop = _mm_andnot_si128(leq_thresh, current);
        __m128i blended = _mm_or_si128(andop, andnop); //abs(p-c) <= thresh ? p : c
        add_low = _mm_unpacklo_epi8(blended, zero);
        add_high = _mm_unpackhi_epi8(blended, zero);
      }

      low = _mm_adds_epu16(low, add_low);
      high = _mm_adds_epu16(high, add_high);
    }

    __m128i acc;
    if (hasSSSE3) {
      // SSSE3: _mm_mulhrs_epi16: r0 := INT16(((a0 * b0) + 0x4000) >> 15)
      low = _mm_mulhrs_epi16(low, div_vector);
      high = _mm_mulhrs_epi16(high, div_vector);
    }
    else {
      // (x*2 * 65536/N + 65536) / 65536 / 2
      // Hi16(x*2 * 65536/N + 1) >> 1
      low = _mm_mulhi_epu16(_mm_slli_epi16(low, 1), div_vector);
      low = _mm_adds_epu16(low, halfdiv_vector);
      low = _mm_srli_epi16(low, 1);
      high = _mm_mulhi_epu16(_mm_slli_epi16(high, 1), div_vector);
      high = _mm_adds_epu16(high, halfdiv_vector);
      high = _mm_srli_epi16(high, 1);
    }
    acc = _mm_packus_epi16(low, high);

    /* old, slowish
    __m128i low_low   = ts_multiply_repack_sse2(_mm_unpacklo_epi16(low, zero), div_vector, halfdiv_vector, zero);
    __m128i low_high  = ts_multiply_repack_sse2(_mm_unpackhi_epi16(low, zero), div_vector, halfdiv_vector, zero);
    __m128i high_low  = ts_multiply_repack_sse2(_mm_unpacklo_epi16(high, zero), div_vector, halfdiv_vector, zero);
    __m128i high_high = ts_multiply_repack_sse2(_mm_unpackhi_epi16(high, zero), div_vector, halfdiv_vector, zero);

    low = _mm_unpacklo_epi32(low_low, low_high);
    high = _mm_unpacklo_epi32(high_low, high_high);
    __m128i acc = _mm_unpacklo_epi64(low, high);
    */

    _mm_store_si128(reinterpret_cast<__m128i*>(c_plane+x), acc);
  }
}


// fast: maxThreshold (255) simple accumulate for average
template<bool maxThreshold, bool hasSSE4, bool lessThan16bit>
static void accumulate_line_16_sse2(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize, int threshold, int div, int bits_per_pixel) {
  // threshold:
  // 10-16 bits: orig threshold scaled by (bits_per_pixel-8)
  int max_pixel_value = (1 << bits_per_pixel) - 1;
  __m128i limit = _mm_set1_epi16(max_pixel_value); //used for clamping when 10-14 bits
  // halfdiv_vector = _mm_set1_epi32(1); // n/a
  __m128 div_vector = _mm_set1_ps(1.0f / (planes + 1));
  __m128i thresh = _mm_set1_epi16(threshold);


  for (size_t x = 0; x < rowsize; x+=16) {
    __m128i current = _mm_load_si128(reinterpret_cast<const __m128i*>(c_plane+x));
    __m128i zero = _mm_setzero_si128();
    __m128i low, high;
    low = _mm_unpacklo_epi16(current, zero);
    high = _mm_unpackhi_epi16(current, zero);

    for (int plane = planes - 1; plane >= 0; --plane) {
      __m128i p = _mm_load_si128(reinterpret_cast<const __m128i*>(planeP[plane] + x));

      __m128i add_low, add_high;
      if (maxThreshold) {
        // fast: simple accumulate for average
        add_low = _mm_unpacklo_epi16(p, zero);
        add_high = _mm_unpackhi_epi16(p, zero);
      } else {
        auto pc = _mm_subs_epu16(p, current); // r2507-
        auto cp = _mm_subs_epu16(current, p);
        auto abs_cp = _mm_or_si128(pc, cp);
        auto leq_thresh = _mm_cmple_epu16<hasSSE4>(abs_cp, thresh);
        /*
        __m128i p_greater_t = _mm_subs_epu16(p, thresh);
        __m128i c_greater_t = _mm_subs_epu16(current, thresh);
        __m128i over_thresh = _mm_or_si128(p_greater_t, c_greater_t); //abs(p-c) - t == (satsub(p,c) | satsub(c,p)) - t =kinda= satsub(p,t) | satsub(c,t)

        __m128i leq_thresh = _mm_cmpeq_epi16(over_thresh, zero); //abs diff lower or equal to threshold
        */

        __m128i andop = _mm_and_si128(leq_thresh, p);
        __m128i andnop = _mm_andnot_si128(leq_thresh, current);
        __m128i blended = _mm_or_si128(andop, andnop); //abs(p-c) <= thresh ? p : c

        add_low = _mm_unpacklo_epi16(blended, zero);
        add_high = _mm_unpackhi_epi16(blended, zero);
      }
      low = _mm_add_epi32(low, add_low);
      high = _mm_add_epi32(high, add_high);
    }

    __m128i acc;
    //__m128 half = _mm_set1_ps(0.5f); // no need rounder, _mm_cvtps_epi32 default is round-to-nearest, unless we use _mm_cvttps_epi32 which truncates
    low = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(low), div_vector));
    high = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(high), div_vector));
    if (hasSSE4)
      acc = _mm_packus_epi32(low, high); // sse4
    else
      acc = _MM_PACKUS_EPI32(low, high);
    if (lessThan16bit)
      if(hasSSE4)
        acc = _mm_min_epu16(acc, limit);
      else
        acc = _MM_MIN_EPU16(acc, limit);

    _mm_store_si128(reinterpret_cast<__m128i*>(c_plane+x), acc);
  }
}

#ifdef X86_32

static __forceinline __m64 ts_multiply_repack_mmx(const __m64 &src, const __m64 &div, __m64 &halfdiv, __m64 &zero) {
  __m64 acc = _mm_madd_pi16(src, div);
  acc = _mm_add_pi32(acc, halfdiv);
  acc = _mm_srli_pi32(acc, 15);
  acc = _mm_packs_pi32(acc, acc);
  return _mm_packs_pu16(acc, zero);
}

//thresh and div must always be 16-bit integers. Thresh is 2 packed bytes and div is a single 16-bit number
static void accumulate_line_mmx(BYTE* c_plane, const BYTE** planeP, int planes, size_t width, int threshold, int div) {
  __m64 halfdiv_vector = _mm_set1_pi32(16384);
  __m64 div_vector = _mm_set1_pi16(div);

  for (size_t x = 0; x < width; x+=8) {
    __m64 current = *reinterpret_cast<const __m64*>(c_plane+x);
    __m64 zero = _mm_setzero_si64();
    __m64 low = _mm_unpacklo_pi8(current, zero);
    __m64 high = _mm_unpackhi_pi8(current, zero);
    __m64 thresh = _mm_set1_pi16(threshold);

    for(int plane = planes-1; plane >= 0; --plane) {
      __m64 p = *reinterpret_cast<const __m64*>(planeP[plane]+x);

      __m64 p_greater_t = _mm_subs_pu8(p, thresh);
      __m64 c_greater_t = _mm_subs_pu8(current, thresh);
      __m64 over_thresh = _mm_or_si64(p_greater_t, c_greater_t); //abs(p-c) - t == (satsub(p,c) | satsub(c,p)) - t =kinda= satsub(p,t) | satsub(c,t)

      __m64 leq_thresh = _mm_cmpeq_pi8(over_thresh, zero); //abs diff lower or equal to threshold

      __m64 andop = _mm_and_si64(leq_thresh, p);
      __m64 andnop = _mm_andnot_si64(leq_thresh, current);
      __m64 blended = _mm_or_si64(andop, andnop); //abs(p-c) <= thresh ? p : c

      __m64 add_low = _mm_unpacklo_pi8(blended, zero);
      __m64 add_high = _mm_unpackhi_pi8(blended, zero);

      low = _mm_adds_pu16(low, add_low);
      high = _mm_adds_pu16(high, add_high);
    }

    __m64 low_low   = ts_multiply_repack_mmx(_mm_unpacklo_pi16(low, zero), div_vector, halfdiv_vector, zero);
    __m64 low_high  = ts_multiply_repack_mmx(_mm_unpackhi_pi16(low, zero), div_vector, halfdiv_vector, zero);
    __m64 high_low  = ts_multiply_repack_mmx(_mm_unpacklo_pi16(high, zero), div_vector, halfdiv_vector, zero);
    __m64 high_high = ts_multiply_repack_mmx(_mm_unpackhi_pi16(high, zero), div_vector, halfdiv_vector, zero);

    low = _mm_unpacklo_pi16(low_low, low_high);
    high = _mm_unpacklo_pi16(high_low, high_high);

   __m64 acc = _mm_unpacklo_pi32(low, high);

    *reinterpret_cast<__m64*>(c_plane+x) = acc;
  }
  _mm_empty();
}

#endif

static void accumulate_line_yuy2(BYTE* c_plane, const BYTE** planeP, int planes, size_t width, BYTE threshold_luma, BYTE threshold_chroma, int div, bool aligned16, IScriptEnvironment* env) {
  if ((env->GetCPUFlags() & CPUF_SSE2) && aligned16 && width >= 16) {
    accumulate_line_sse2<false, false>(c_plane, planeP, planes, width, threshold_luma | (threshold_chroma << 8), div);
  } else
#ifdef X86_32
  if ((env->GetCPUFlags() & CPUF_MMX) && width >= 8) {
    accumulate_line_mmx(c_plane, planeP, planes, width, threshold_luma | (threshold_chroma << 8), div); //yuy2 is always at least mod8
  } else
#endif
    accumulate_line_yuy2_c(c_plane, planeP, planes, width, threshold_luma, threshold_chroma, div);
}

static void accumulate_line(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize, BYTE threshold, int div, bool aligned16, int pixelsize, int bits_per_pixel, IScriptEnvironment* env) {
  // todo SSE2 float
  // threshold == 255: simple average
  bool maxThreshold = (threshold == 255);
  if ((pixelsize == 2) && (env->GetCPUFlags() & CPUF_SSE4) && aligned16 && rowsize >= 16) {
    // <maxThreshold, hasSSE4, lessThan16bit>
    if(maxThreshold) {
      if(bits_per_pixel < 16)
        accumulate_line_16_sse2<true, true, true>(c_plane, planeP, planes, rowsize, threshold << (bits_per_pixel - 8), div, bits_per_pixel);
      else
        accumulate_line_16_sse2<true, true, false>(c_plane, planeP, planes, rowsize, threshold << (bits_per_pixel - 8), div, bits_per_pixel);
    }
    else {
      if (bits_per_pixel < 16)
        accumulate_line_16_sse2<false, true, true>(c_plane, planeP, planes, rowsize, threshold << (bits_per_pixel - 8), div, bits_per_pixel);
      else
        accumulate_line_16_sse2<false, true, false>(c_plane, planeP, planes, rowsize, threshold << (bits_per_pixel - 8), div, bits_per_pixel);
    }
  } else if ((pixelsize == 2) && (env->GetCPUFlags() & CPUF_SSE2) && aligned16 && rowsize >= 16) {
    // <maxThreshold, hasSSE4, lessThan16bit>
    if(maxThreshold) {
      if(bits_per_pixel < 16)
        accumulate_line_16_sse2<true, false, true>(c_plane, planeP, planes, rowsize, threshold << (bits_per_pixel - 8), div, bits_per_pixel);
      else
        accumulate_line_16_sse2<true, false, false>(c_plane, planeP, planes, rowsize, threshold << (bits_per_pixel - 8), div, bits_per_pixel);
    }
    else {
      if (bits_per_pixel < 16)
        accumulate_line_16_sse2<false, false, true>(c_plane, planeP, planes, rowsize, threshold << (bits_per_pixel - 8), div, bits_per_pixel);
      else
        accumulate_line_16_sse2<false, false, false>(c_plane, planeP, planes, rowsize, threshold << (bits_per_pixel - 8), div, bits_per_pixel);
    }
  }
  else if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSSE3) && aligned16 && rowsize >= 16) {
    if (maxThreshold) // <maxThreshold, hasSSSE3
      accumulate_line_sse2<true, true>(c_plane, planeP, planes, rowsize, threshold | (threshold << 8), div);
    else
      accumulate_line_sse2<false, true>(c_plane, planeP, planes, rowsize, threshold | (threshold << 8), div);
  } else if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSE2) && aligned16 && rowsize >= 16) {
    if (maxThreshold)
      accumulate_line_sse2<true, false>(c_plane, planeP, planes, rowsize, threshold | (threshold << 8), div);
    else
      accumulate_line_sse2<false, false>(c_plane, planeP, planes, rowsize, threshold | (threshold << 8), div);
  }
  else
#ifdef X86_32
  if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_MMX) && rowsize >= 8) {
    size_t mod8_width = rowsize / 8 * 8;
    accumulate_line_mmx(c_plane, planeP, planes, rowsize, threshold | (threshold << 8), div);

    if (mod8_width != rowsize) {
      accumulate_line_c<uint8_t, false>(c_plane, planeP, planes, mod8_width, rowsize - mod8_width, threshold, div, bits_per_pixel);
    }
  } else
#endif
    switch(pixelsize) {
    case 1:
      if (maxThreshold)
        accumulate_line_c<uint8_t, true>(c_plane, planeP, planes, 0, rowsize, threshold, div, bits_per_pixel);
      else
        accumulate_line_c<uint8_t, false>(c_plane, planeP, planes, 0, rowsize, threshold, div, bits_per_pixel);
      break;
    case 2:
      if (maxThreshold)
        accumulate_line_c<uint16_t, true>(c_plane, planeP, planes, 0, rowsize, threshold, div, bits_per_pixel);
      else
        accumulate_line_c<uint16_t, false>(c_plane, planeP, planes, 0, rowsize, threshold, div, bits_per_pixel);
      break;
    case 4:
      if (maxThreshold)
        accumulate_line_c<float, true>(c_plane, planeP, planes, 0, rowsize, threshold, div, bits_per_pixel);
      else
        accumulate_line_c<float, false>(c_plane, planeP, planes, 0, rowsize, threshold, div, bits_per_pixel);
      break;
    }
}

// may also used from conditionalfunctions
// packed rgb template masks out alpha plane for RGB32
template<bool packedRGB3264>
int calculate_sad_sse2(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height)
{
  size_t mod16_width = rowsize / 16 * 16;
  int result = 0;
  __m128i sum = _mm_setzero_si128();

  __m128i rgb_mask;
  if (packedRGB3264) {
    rgb_mask = _mm_set1_epi32(0x00FFFFFF);
  }

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod16_width; x+=16) {
      __m128i cur = _mm_load_si128(reinterpret_cast<const __m128i*>(cur_ptr + x));
      __m128i other = _mm_load_si128(reinterpret_cast<const __m128i*>(other_ptr + x));
      if (packedRGB3264) {
        cur = _mm_and_si128(cur, rgb_mask);  // mask out A channel
        other = _mm_and_si128(other, rgb_mask);
      }
      __m128i sad = _mm_sad_epu8(cur, other);
      sum = _mm_add_epi32(sum, sad);
    }
    if (mod16_width != rowsize) {
      if (packedRGB3264)
        for (size_t x = mod16_width / 4; x < rowsize / 4; x += 4) {
          result += std::abs(cur_ptr[x*4+0] - other_ptr[x*4+0]) +
            std::abs(cur_ptr[x*4+1] - other_ptr[x*4+1]) +
            std::abs(cur_ptr[x*4+2] - other_ptr[x*4+2]);
          // no alpha
        }
      else
        for (size_t x = mod16_width; x < rowsize; ++x) {
          result += std::abs(cur_ptr[x] - other_ptr[x]);
        }
    }
    cur_ptr += cur_pitch;
    other_ptr += other_pitch;
  }
  __m128i upper = _mm_castps_si128(_mm_movehl_ps(_mm_setzero_ps(), _mm_castsi128_ps(sum)));
  sum = _mm_add_epi32(sum, upper);
  result += _mm_cvtsi128_si32(sum);
  return result;
}

// instantiate
template int calculate_sad_sse2<false>(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height);
template int calculate_sad_sse2<true>(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height);


// works for uint8_t, but there is a specific, bit faster function above
// also used from conditionalfunctions
// packed rgb template masks out alpha plane for RGB32/RGB64
template<typename pixel_t, bool packedRGB3264>
__int64 calculate_sad_8_or_16_sse2(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height)
{
  size_t mod16_width = rowsize / 16 * 16;

  __m128i zero = _mm_setzero_si128();
  __int64 totalsum = 0; // fullframe SAD exceeds int32 at 8+ bit

  __m128i rgb_mask;
  if (packedRGB3264) {
    if (sizeof(pixel_t) == 1)
      rgb_mask = _mm_set1_epi32(0x00FFFFFF);
    else
      rgb_mask = _mm_set_epi32(0x0000FFFF,0xFFFFFFFF,0x0000FFFF,0xFFFFFFFF);
  }

  for ( size_t y = 0; y < height; y++ )
  {
    __m128i sum = _mm_setzero_si128(); // for one row int is enough
    for ( size_t x = 0; x < mod16_width; x+=16 )
    {
      __m128i src1, src2;
      src1 = _mm_load_si128((__m128i *) (cur_ptr + x));   // 16 bytes or 8 words
      src2 = _mm_load_si128((__m128i *) (other_ptr + x));
      if (packedRGB3264) {
        src1 = _mm_and_si128(src1, rgb_mask); // mask out A channel
        src2 = _mm_and_si128(src2, rgb_mask);
      }
      if(sizeof(pixel_t) == 1) {
        // this is uint_16 specific, but leave here for sample
        sum = _mm_add_epi32(sum, _mm_sad_epu8(src1, src2)); // sum0_32, 0, sum1_32, 0
      }
      else if (sizeof(pixel_t) == 2) {
        __m128i greater_t = _mm_subs_epu16(src1, src2); // unsigned sub with saturation
        __m128i smaller_t = _mm_subs_epu16(src2, src1);
        __m128i absdiff = _mm_or_si128(greater_t, smaller_t); //abs(s1-s2)  == (satsub(s1,s2) | satsub(s2,s1))
        // 8 x uint16 absolute differences
        sum = _mm_add_epi32(sum, _mm_unpacklo_epi16(absdiff, zero));
        sum = _mm_add_epi32(sum, _mm_unpackhi_epi16(absdiff, zero));
        // sum0_32, sum1_32, sum2_32, sum3_32
      }
    }
    // summing up partial sums
    if(sizeof(pixel_t) == 2) {
      // at 16 bits: we have 4 integers for sum: a0 a1 a2 a3
      __m128i a0_a1 = _mm_unpacklo_epi32(sum, zero); // a0 0 a1 0
      __m128i a2_a3 = _mm_unpackhi_epi32(sum, zero); // a2 0 a3 0
      sum = _mm_add_epi32( a0_a1, a2_a3 ); // a0+a2, 0, a1+a3, 0
      /* SSSE3: told to be not too fast
      sum = _mm_hadd_epi32(sum, zero);  // A1+A2, B1+B2, 0+0, 0+0
      sum = _mm_hadd_epi32(sum, zero);  // A1+A2+B1+B2, 0+0+0+0, 0+0+0+0, 0+0+0+0
      */
    }

    // sum here: two 32 bit partial result: sum1 0 sum2 0
    __m128i sum_hi = _mm_unpackhi_epi64(sum, zero);
    // or: __m128i sum_hi = _mm_castps_si128(_mm_movehl_ps(_mm_setzero_ps(), _mm_castsi128_ps(sum)));
    sum = _mm_add_epi32(sum, sum_hi);
    int rowsum = _mm_cvtsi128_si32(sum);

    // rest
    if (mod16_width != rowsize) {
      if (packedRGB3264)
        for (size_t x = mod16_width / sizeof(pixel_t) / 4; x < rowsize / sizeof(pixel_t) / 4; x += 4) {
          rowsum += std::abs(reinterpret_cast<const pixel_t *>(cur_ptr)[x*4+0] - reinterpret_cast<const pixel_t *>(other_ptr)[x*4+0]) +
            std::abs(reinterpret_cast<const pixel_t *>(cur_ptr)[x*4+1] - reinterpret_cast<const pixel_t *>(other_ptr)[x*4+1]) +
            std::abs(reinterpret_cast<const pixel_t *>(cur_ptr)[x*4+2] - reinterpret_cast<const pixel_t *>(other_ptr)[x*4+2]);
          // no alpha
        }
      else
        for (size_t x = mod16_width / sizeof(pixel_t); x < rowsize / sizeof(pixel_t); ++x) {
          rowsum += std::abs(reinterpret_cast<const pixel_t *>(cur_ptr)[x] - reinterpret_cast<const pixel_t *>(other_ptr)[x]);
        }
    }

    totalsum += rowsum;

    cur_ptr += cur_pitch;
    other_ptr += other_pitch;
  }
  return totalsum;
}

// instantiate
template __int64 calculate_sad_8_or_16_sse2<uint8_t, false>(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height);
template __int64 calculate_sad_8_or_16_sse2<uint8_t, true>(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height);
template __int64 calculate_sad_8_or_16_sse2<uint16_t, false>(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height);
template __int64 calculate_sad_8_or_16_sse2<uint16_t, true>(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height);


#ifdef X86_32
static int calculate_sad_isse(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height)
{
  size_t mod8_width = rowsize / 8 * 8;
  int result = 0;
  __m64 sum = _mm_setzero_si64();
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod8_width; x+=8) {
      __m64 cur = *reinterpret_cast<const __m64*>(cur_ptr + x);
      __m64 other = *reinterpret_cast<const __m64*>(other_ptr + x);
      __m64 sad = _mm_sad_pu8(cur, other);
      sum = _mm_add_pi32(sum, sad);
    }
    if (mod8_width != rowsize) {
      for (size_t x = mod8_width; x < rowsize; ++x) {
        result += std::abs(cur_ptr[x] - other_ptr[x]);
      }
    }

    cur_ptr += cur_pitch;
    other_ptr += other_pitch;
  }
  result += _mm_cvtsi64_si32(sum);
  _mm_empty();
  return result;
}
#endif

template<typename pixel_t>
static __int64 calculate_sad_c(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height)
{
  const pixel_t *ptr1 = reinterpret_cast<const pixel_t *>(cur_ptr);
  const pixel_t *ptr2 = reinterpret_cast<const pixel_t *>(other_ptr);
  size_t width = rowsize / sizeof(pixel_t);
  cur_pitch /= sizeof(pixel_t);
  other_pitch /= sizeof(pixel_t);

  // for fullframe float may loose precision
  typedef typename std::conditional < std::is_floating_point<pixel_t>::value, double, __int64>::type sum_t;
  // for one row int is enough and faster than int64
  typedef typename std::conditional < std::is_floating_point<pixel_t>::value, float, int>::type sumrow_t;
  sum_t sum = 0;

  for (size_t y = 0; y < height; ++y) {
    sumrow_t sumrow = 0;
    for (size_t x = 0; x < width; ++x) {
      sumrow += std::abs(ptr1[x] - ptr2[x]);
    }
    sum += sumrow;
    ptr1 += cur_pitch;
    ptr2 += other_pitch;
  }
  if (std::is_floating_point<pixel_t>::value)
    return (__int64)(sum * 255); // scale 0..1 based sum to 8 bit range
  else
    return (__int64)sum; // for int, scaling to 8 bit range is done outside
}

// sum of byte-diffs.
static __int64 calculate_sad(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height, int pixelsize, int bits_per_pixel, IScriptEnvironment* env) {
  // todo: sse for float
  if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(cur_ptr, 16) && IsPtrAligned(other_ptr, 16) && rowsize >= 16) {
    return (__int64)calculate_sad_sse2<false>(cur_ptr, other_ptr, cur_pitch, other_pitch, rowsize, height);
  }
#ifdef X86_32
  if ((pixelsize ==1 ) && (env->GetCPUFlags() & CPUF_INTEGER_SSE) && rowsize >= 8) {
    return (__int64)calculate_sad_isse(cur_ptr, other_ptr, cur_pitch, other_pitch, rowsize, height);
  }
#endif
  // sse2 uint16_t
  if ((pixelsize == 2) && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(cur_ptr, 16) && IsPtrAligned(other_ptr, 16) && rowsize >= 16) {
    return calculate_sad_8_or_16_sse2<uint16_t, false>(cur_ptr, other_ptr, cur_pitch, other_pitch, rowsize, height) >> (bits_per_pixel-8);
  }

  switch(pixelsize) {
  case 1: return calculate_sad_c<uint8_t>(cur_ptr, other_ptr, cur_pitch, other_pitch, rowsize, height);
  case 2: return calculate_sad_c<uint16_t>(cur_ptr, other_ptr, cur_pitch, other_pitch, rowsize, height) >> (bits_per_pixel-8); // scale back to 8 bit range;
  default: // case 4
    return calculate_sad_c<float>(cur_ptr, other_ptr, cur_pitch, other_pitch, rowsize, height);
  }
}

PVideoFrame TemporalSoften::GetFrame(int n, IScriptEnvironment* env)
{
  int radius = (kernel-1) / 2;
  int c = 0;

  // Just skip if silly settings

  if ((!luma_threshold && !chroma_threshold) || !radius)
  {
    PVideoFrame ret = child->GetFrame(n, env); // P.F.
    return ret;
  }

  bool planeDisabled[16];

  for (int p = 0; p<16; p++) {
    planeDisabled[p] = false;
  }

  std::vector<PVideoFrame> frames;
  frames.reserve(kernel);

  for (int p = n - radius; p <= n + radius; ++p) {
    frames.emplace_back(child->GetFrame(clamp(p, 0, vi.num_frames - 1), env));
  }

  // P.F. 16.04.06 leak fix r1841 after 8 days of bug chasing:
  // Reason #1 of the random QTGMC memory leaks (stuck frame refcounts)
  // MakeWritable alters the pointer if it is not yet writeable, thus the original PVideoFrame won't be freed (refcount decremented)
  // To fix this, we leave the frame[] array in its place and copy frame[radius] to CenterFrame and make further write operations on this new frame.
  // env->MakeWritable(&frames[radius]); // old culprit line. if not yet writeable -> gives another pointer
  PVideoFrame CenterFrame = frames[radius];
  env->MakeWritable(&CenterFrame);

  do {
    const BYTE* planeP[16];
    const BYTE* planeP2[16];
    int planePitch[16];
    int planePitch2[16];

    int current_thresh = planes[c].threshold;  // Threshold for current plane.
    int d = 0;
    for (int i = 0; i<radius; i++) { // Fetch all planes sequencially
      planePitch[d] = frames[i]->GetPitch(planes[c].planeId);
      planeP[d++] = frames[i]->GetReadPtr(planes[c].planeId);
    }

//    BYTE* c_plane = frames[radius]->GetWritePtr(planes[c]);
    BYTE* c_plane = CenterFrame->GetWritePtr(planes[c].planeId); // P.F. using CenterFrame for write access

    for (int i = 1; i<=radius; i++) { // Fetch all planes sequencially
      planePitch[d] = frames[radius+i]->GetPitch(planes[c].planeId);
      planeP[d++] = frames[radius+i]->GetReadPtr(planes[c].planeId);
    }

    int rowsize = frames[radius]->GetRowSize(planes[c].planeId|PLANAR_ALIGNED);
    int h = frames[radius]->GetHeight(planes[c].planeId);
    int pitch = frames[radius]->GetPitch(planes[c].planeId);

    if (scenechange>0) {
      int d2 = 0;
      bool skiprest = false;
      for (int i = radius-1; i>=0; i--) { // Check frames backwards
        if ((!skiprest) && (!planeDisabled[i])) {
          int sad = (int)calculate_sad(c_plane, planeP[i], pitch, planePitch[i], frames[radius]->GetRowSize(planes[c].planeId), h, pixelsize, bits_per_pixel, env);
          if (sad < scenechange) {
            planePitch2[d2] = planePitch[i];
            planeP2[d2++] = planeP[i];
          } else {
            skiprest = true;
          }
          planeDisabled[i] = skiprest;  // Disable this frame on next plane (so that Y can affect UV)
        } else {
          planeDisabled[i] = true;
        }
      }
      skiprest = false;
      for (int i = radius; i < 2*radius; i++) { // Check forward frames
        if ((!skiprest)  && (!planeDisabled[i])) {   // Disable this frame on next plane (so that Y can affect UV)
          int sad = (int)calculate_sad(c_plane, planeP[i], pitch, planePitch[i], frames[radius]->GetRowSize(planes[c].planeId), h, pixelsize, bits_per_pixel, env);
          if (sad < scenechange) {
            planePitch2[d2] = planePitch[i];
            planeP2[d2++] = planeP[i];
          } else {
            skiprest = true;
          }
          planeDisabled[i] = skiprest;
        } else {
          planeDisabled[i] = true;
        }
      }

      //Copy back
      for (int i = 0; i<d2; i++) {
        planeP[i] = planeP2[i];
        planePitch[i] = planePitch2[i];
      }
      d = d2;
    }

    if (d < 1)
    {
      // Memory leak reason #2 r1841: this wasn't here before return
      for (int i = 0; i < kernel; ++i)
        frames[i] = nullptr;
      // return frames[radius];
      return CenterFrame; // return the modified frame
    }

    int c_div = 32768/(d+1);  // We also have the tetplane included, thus d+1.
    if (current_thresh) {
      bool aligned16 = IsPtrAligned(c_plane, 16);
      if ((env->GetCPUFlags() & CPUF_SSE2) && aligned16) {
        for (int i = 0; i < d; ++i) {
          aligned16 = aligned16 && IsPtrAligned(planeP[i], 16);
        }
      }
      // for threshold==255 -> simple average
      for (int y = 0; y<h; y++) { // One line at the time
        if (vi.IsYUY2()) {
          accumulate_line_yuy2(c_plane, planeP, d, rowsize, luma_threshold, chroma_threshold, c_div, aligned16, env);
        } else {
          accumulate_line(c_plane, planeP, d, rowsize, current_thresh, c_div, aligned16, pixelsize, bits_per_pixel, env);
        }
        for (int p = 0; p<d; p++)
          planeP[p] += planePitch[p];
        c_plane += pitch;
      }
    } else { // Just maintain the plane
    }
    c++;
  } while (planes[c].planeId);

  //  PVideoFrame result = frames[radius]; // we are using CenterFrame instead
  //  return result;
  return CenterFrame;
}


AVSValue __cdecl TemporalSoften::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  return new TemporalSoften( args[0].AsClip(), args[1].AsInt(), args[2].AsInt(),
                             args[3].AsInt(), args[4].AsInt(0),/*args[5].AsInt(1),*/env ); //ignore mode parameter
}



/****************************
 ****  Spatial Soften   *****
 ***************************/

SpatialSoften::SpatialSoften( PClip _child, int _radius, unsigned _luma_threshold,
                              unsigned _chroma_threshold, IScriptEnvironment* env )
  : GenericVideoFilter(_child), diameter(_radius*2+1),
    luma_threshold(_luma_threshold), chroma_threshold(_chroma_threshold)
{
  if (!vi.IsYUY2())
    env->ThrowError("SpatialSoften: requires YUY2 input");
}


PVideoFrame SpatialSoften::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrame(vi);

  const BYTE* srcp = src->GetReadPtr();
  BYTE* dstp = dst->GetWritePtr();
  int src_pitch = src->GetPitch();
  int dst_pitch = dst->GetPitch();
  int row_size = src->GetRowSize();

  for (int y=0; y<vi.height; ++y)
  {
    const BYTE* line[65];    // better not make diameter bigger than this...
    for (int h=0; h<diameter; ++h)
      line[h] = &srcp[src_pitch * clamp(y+h-(diameter>>1), 0, vi.height-1)];
    int x;

    int edge = (diameter+1) & -4;
    for (x=0; x<edge; ++x)  // diameter-1 == (diameter>>1) * 2
      dstp[y*dst_pitch + x] = srcp[y*src_pitch + x];
    for (; x < row_size - edge; x+=2)
    {
      int cnt=0, _y=0, _u=0, _v=0;
      int xx = x | 3;
      int Y = srcp[y*src_pitch + x], U = srcp[y*src_pitch + xx - 2], V = srcp[y*src_pitch + xx];
      for (int h=0; h<diameter; ++h)
      {
        for (int w = -diameter+1; w < diameter; w += 2)
        {
          int xw = (x+w) | 3;
          if (IsClose(line[h][x+w], Y, luma_threshold) && IsClose(line[h][xw-2], U,
                      chroma_threshold) && IsClose(line[h][xw], V, chroma_threshold))
          {
            ++cnt; _y += line[h][x+w]; _u += line[h][xw-2]; _v += line[h][xw];
          }
        }
      }
      dstp[y*dst_pitch + x] = (_y + (cnt>>1)) / cnt;
      if (!(x&3)) {
        dstp[y*dst_pitch + x+1] = (_u + (cnt>>1)) / cnt;
        dstp[y*dst_pitch + x+3] = (_v + (cnt>>1)) / cnt;
      }
    }
    for (; x<row_size; ++x)
      dstp[y*dst_pitch + x] = srcp[y*src_pitch + x];
  }

  return dst;
}


AVSValue __cdecl SpatialSoften::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  return new SpatialSoften( args[0].AsClip(), args[1].AsInt(), args[2].AsInt(),
                            args[3].AsInt(), env );
}
