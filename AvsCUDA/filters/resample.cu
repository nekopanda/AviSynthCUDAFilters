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

#include "../AvsCUDA.h"
#include "resample.h"
#include "resample_avx2.h"
#include <avs/config.h>
#include "../core/internal.h"

#include "turn.h"
#include <avs/alignment.h>

#include <type_traits>
// Intrinsics for SSE4.1, SSSE3, SSE3, SSE2, ISSE and MMX
#include <emmintrin.h>
#include <smmintrin.h>
#include <algorithm>

/***************************************
********* Templated SSE Loader ********
***************************************/

typedef __m128i (SSELoader)(const __m128i*);
typedef __m128 (SSELoader_ps)(const float*);

__forceinline __m128i simd_load_aligned(const __m128i* adr)
{
	return _mm_load_si128(adr);
}

__forceinline __m128i simd_load_unaligned(const __m128i* adr)
{
	return _mm_loadu_si128(adr);
}

__forceinline __m128i simd_load_unaligned_sse3(const __m128i* adr)
{
	return _mm_lddqu_si128(adr);
}

__forceinline __m128i simd_load_streaming(const __m128i* adr)
{
	return _mm_stream_load_si128(const_cast<__m128i*>(adr));
}

// float loaders
__forceinline __m128 simd_loadps_aligned(const float * adr)
{
	return _mm_load_ps(adr);
}

__forceinline __m128 simd_loadps_unaligned(const float* adr)
{
	return _mm_loadu_ps(adr);
}


/***************************************
***** Vertical Resizer Assembly *******
***************************************/

template<typename pixel_t>
static void resize_v_planar_pointresize(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  AVS_UNUSED(src_pitch);
  AVS_UNUSED(bits_per_pixel);
  AVS_UNUSED(storage);

	pixel_t* src0 = (pixel_t *)src;
	pixel_t* dst0 = (pixel_t *)dst;
	dst_pitch = dst_pitch / sizeof(pixel_t);

	for (int y = 0; y < target_height; y++) {
		int offset = program->pixel_offset[y];
		const pixel_t* src_ptr = src0 + pitch_table[offset] / sizeof(pixel_t);

		memcpy(dst0, src_ptr, width * sizeof(pixel_t));

		dst0 += dst_pitch;
	}
}

template<typename pixel_t>
__global__ void kl_resize_v_planar_pointresize(
	pixel_t* dst, const pixel_t* __restrict__ src, int dst_pitch, int src_pitch,
	const int* __restrict__ pixel_offset,
	int target_width, int target_height, float limit, int filter_size)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < target_width && y < target_height) {
		dst[x + y * dst_pitch] = src[x + pixel_offset[y] * src_pitch];
	}
}

template<typename pixel_t>
void launch_resize_v_planar_pointresize(
	BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch,
	const int* pixel_offset, const float* pixel_coefficient,
	int target_width, int target_height, float limit, int filter_size)
{
	dim3 threads(32, 16);
	dim3 blocks(nblocks(target_width, threads.x), nblocks(target_height, threads.y));

	kl_resize_v_planar_pointresize << <blocks, threads >> > (
		(pixel_t*)dst, (const pixel_t*)src, dst_pitch / sizeof(pixel_t), src_pitch / sizeof(pixel_t),
		pixel_offset, target_width, target_height, limit, filter_size);
}

template<typename pixel_t>
static void resize_v_c_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  AVS_UNUSED(src_pitch);
  AVS_UNUSED(storage);
	int filter_size = program->filter_size;

	typedef typename std::conditional < std::is_floating_point<pixel_t>::value, float, short>::type coeff_t;
	coeff_t *current_coeff;

	if (!std::is_floating_point<pixel_t>::value)
		current_coeff = (coeff_t *)program->pixel_coefficient;
	else
		current_coeff = (coeff_t *)program->pixel_coefficient_float;

	pixel_t* src0 = (pixel_t *)src;
	pixel_t* dst0 = (pixel_t *)dst;
	dst_pitch = dst_pitch / sizeof(pixel_t);

	pixel_t limit = 0;
	if (!std::is_floating_point<pixel_t>::value) {  // floats are unscaled and uncapped
    if constexpr(sizeof(pixel_t) == 1) limit = 255;
    else if constexpr(sizeof(pixel_t) == 2) limit = pixel_t((1 << bits_per_pixel) - 1);
	}

	for (int y = 0; y < target_height; y++) {
		int offset = program->pixel_offset[y];
		const pixel_t* src_ptr = src0 + pitch_table[offset] / sizeof(pixel_t);

		for (int x = 0; x < width; x++) {
			// todo: check whether int result is enough for 16 bit samples (can an int overflow because of 16384 scale or really need __int64?)
			typename std::conditional < sizeof(pixel_t) == 1, int, typename std::conditional < sizeof(pixel_t) == 2, __int64, float>::type >::type result;
			result = 0;
			for (int i = 0; i < filter_size; i++) {
				result += (src_ptr + pitch_table[i] / sizeof(pixel_t))[x] * current_coeff[i];
			}
			if (!std::is_floating_point<pixel_t>::value) {  // floats are unscaled and uncapped
        if constexpr(sizeof(pixel_t) == 1)
					result = (result + (1 << (FPScale8bits - 1))) / (1 << FPScale8bits);
        else if constexpr(sizeof(pixel_t) == 2)
					result = (result + (1 << (FPScale16bits - 1))) / (1 << FPScale16bits);
				result = clamp(result, decltype(result)(0), decltype(result)(limit));
			}
			dst0[x] = (pixel_t)result;
		}

		dst0 += dst_pitch;
		current_coeff += filter_size;
	}
}

template<typename pixel_t>
__global__ void kl_resize_v_planar(
	pixel_t* dst, const pixel_t* __restrict__ src, int dst_pitch, int src_pitch,
	const int* __restrict__ pixel_offset, const float* __restrict__ pixel_coefficient,
	int target_width, int target_height, float limit, int filter_size)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < target_width && y < target_height) {
		float result = 0;
		int y_offset = pixel_offset[y];
		for (int i = 0; i < filter_size; ++i) {
			result += pixel_coefficient[i] * src[x + (y_offset + i) * src_pitch];
		}
		dst[x + y * dst_pitch] = (pixel_t)clamp<float>(result, 0, limit);
	}
}

template<typename pixel_t>
void launch_resize_v_planar(
	BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch,
	const int* pixel_offset, const float* pixel_coefficient,
	int target_width, int target_height, float limit, int filter_size)
{
	dim3 threads(32, 16);
	dim3 blocks(nblocks(target_width, threads.x), nblocks(target_height, threads.y));

	kl_resize_v_planar << <blocks, threads >> > (
		(pixel_t*)dst, (const pixel_t*)src, dst_pitch / sizeof(pixel_t), src_pitch / sizeof(pixel_t),
		pixel_offset, pixel_coefficient, target_width, target_height, limit, filter_size);
}

#ifdef X86_32
static void resize_v_mmx_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
	int filter_size = program->filter_size;
	short* current_coeff = program->pixel_coefficient;

	int wMod8 = (width / 8) * 8;
	int sizeMod2 = (filter_size / 2) * 2;
	bool notMod2 = sizeMod2 < filter_size;

	__m64 zero = _mm_setzero_si64();

	for (int y = 0; y < target_height; y++) {
		int offset = program->pixel_offset[y];
		const BYTE* src_ptr = src + pitch_table[offset];

		for (int x = 0; x < wMod8; x += 8) {
			__m64 result_1 = _mm_set1_pi32(8192); // Init. with rounder (16384/2 = 8192)
			__m64 result_2 = result_1;
			__m64 result_3 = result_1;
			__m64 result_4 = result_1;

			for (int i = 0; i < sizeMod2; i += 2) {
				__m64 src_p1 = *(reinterpret_cast<const __m64*>(src_ptr + pitch_table[i] + x));   // For detailed explanation please see SSE2 version.
				__m64 src_p2 = *(reinterpret_cast<const __m64*>(src_ptr + pitch_table[i + 1] + x));

				__m64 src_l = _mm_unpacklo_pi8(src_p1, src_p2);
				__m64 src_h = _mm_unpackhi_pi8(src_p1, src_p2);

				__m64 src_1 = _mm_unpacklo_pi8(src_l, zero);
				__m64 src_2 = _mm_unpackhi_pi8(src_l, zero);
				__m64 src_3 = _mm_unpacklo_pi8(src_h, zero);
				__m64 src_4 = _mm_unpackhi_pi8(src_h, zero);

				__m64 coeff = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(current_coeff + i));
				coeff = _mm_unpacklo_pi32(coeff, coeff);

				__m64 dst_1 = _mm_madd_pi16(src_1, coeff);
				__m64 dst_2 = _mm_madd_pi16(src_2, coeff);
				__m64 dst_3 = _mm_madd_pi16(src_3, coeff);
				__m64 dst_4 = _mm_madd_pi16(src_4, coeff);

				result_1 = _mm_add_pi32(result_1, dst_1);
				result_2 = _mm_add_pi32(result_2, dst_2);
				result_3 = _mm_add_pi32(result_3, dst_3);
				result_4 = _mm_add_pi32(result_4, dst_4);
			}

			if (notMod2) { // do last odd row
				__m64 src_p = *(reinterpret_cast<const __m64*>(src_ptr + pitch_table[sizeMod2] + x));

				__m64 src_l = _mm_unpacklo_pi8(src_p, zero);
				__m64 src_h = _mm_unpackhi_pi8(src_p, zero);

				__m64 coeff = _mm_set1_pi16(current_coeff[sizeMod2]);

				__m64 dst_ll = _mm_mullo_pi16(src_l, coeff);   // Multiply by coefficient
				__m64 dst_lh = _mm_mulhi_pi16(src_l, coeff);
				__m64 dst_hl = _mm_mullo_pi16(src_h, coeff);
				__m64 dst_hh = _mm_mulhi_pi16(src_h, coeff);

				__m64 dst_1 = _mm_unpacklo_pi16(dst_ll, dst_lh); // Unpack to 32-bit integer
				__m64 dst_2 = _mm_unpackhi_pi16(dst_ll, dst_lh);
				__m64 dst_3 = _mm_unpacklo_pi16(dst_hl, dst_hh);
				__m64 dst_4 = _mm_unpackhi_pi16(dst_hl, dst_hh);

				result_1 = _mm_add_pi32(result_1, dst_1);
				result_2 = _mm_add_pi32(result_2, dst_2);
				result_3 = _mm_add_pi32(result_3, dst_3);
				result_4 = _mm_add_pi32(result_4, dst_4);
			}

			// Divide by 16348 (FPRound)
			result_1 = _mm_srai_pi32(result_1, 14);
			result_2 = _mm_srai_pi32(result_2, 14);
			result_3 = _mm_srai_pi32(result_3, 14);
			result_4 = _mm_srai_pi32(result_4, 14);

			// Pack and store
			__m64 result_l = _mm_packs_pi32(result_1, result_2);
			__m64 result_h = _mm_packs_pi32(result_3, result_4);
			__m64 result = _mm_packs_pu16(result_l, result_h);

			*(reinterpret_cast<__m64*>(dst + x)) = result;
		}

		// Leftover
		for (int x = wMod8; x < width; x++) {
			int result = 0;
			for (int i = 0; i < filter_size; i++) {
				result += (src_ptr + pitch_table[i])[x] * current_coeff[i];
			}
			result = ((result + 8192) / 16384);
			result = result > 255 ? 255 : result < 0 ? 0 : result;
			dst[x] = (BYTE)result;
		}

		dst += dst_pitch;
		current_coeff += filter_size;
	}

	_mm_empty();
}
#endif

template<SSELoader load>
static void resize_v_sse2_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  AVS_UNUSED(src_pitch);
  AVS_UNUSED(bits_per_pixel);
  AVS_UNUSED(storage);

	int filter_size = program->filter_size;
	short* current_coeff = program->pixel_coefficient;

	int wMod16 = (width / 16) * 16;
	int sizeMod2 = (filter_size / 2) * 2;
	bool notMod2 = sizeMod2 < filter_size;

	__m128i zero = _mm_setzero_si128();

	for (int y = 0; y < target_height; y++) {
		int offset = program->pixel_offset[y];
		const BYTE* src_ptr = src + pitch_table[offset];

		for (int x = 0; x < wMod16; x += 16) {
			__m128i result_1 = _mm_set1_epi32(8192); // Init. with rounder (16384/2 = 8192)
			__m128i result_2 = result_1;
			__m128i result_3 = result_1;
			__m128i result_4 = result_1;

			for (int i = 0; i < sizeMod2; i += 2) {
				__m128i src_p1 = load(reinterpret_cast<const __m128i*>(src_ptr + pitch_table[i] + x));   // p|o|n|m|l|k|j|i|h|g|f|e|d|c|b|a
				__m128i src_p2 = load(reinterpret_cast<const __m128i*>(src_ptr + pitch_table[i + 1] + x)); // P|O|N|M|L|K|J|I|H|G|F|E|D|C|B|A

				__m128i src_l = _mm_unpacklo_epi8(src_p1, src_p2);                                   // Hh|Gg|Ff|Ee|Dd|Cc|Bb|Aa
				__m128i src_h = _mm_unpackhi_epi8(src_p1, src_p2);                                   // Pp|Oo|Nn|Mm|Ll|Kk|Jj|Ii

				__m128i src_1 = _mm_unpacklo_epi8(src_l, zero);                                      // .D|.d|.C|.c|.B|.b|.A|.a
				__m128i src_2 = _mm_unpackhi_epi8(src_l, zero);                                      // .H|.h|.G|.g|.F|.f|.E|.e
				__m128i src_3 = _mm_unpacklo_epi8(src_h, zero);                                      // etc.
				__m128i src_4 = _mm_unpackhi_epi8(src_h, zero);                                      // etc.

				__m128i coeff = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(current_coeff + i));   // XX|XX|XX|XX|XX|XX|CO|co
				coeff = _mm_shuffle_epi32(coeff, 0);                                                 // CO|co|CO|co|CO|co|CO|co

				__m128i dst_1 = _mm_madd_epi16(src_1, coeff);                                         // CO*D+co*d | CO*C+co*c | CO*B+co*b | CO*A+co*a
				__m128i dst_2 = _mm_madd_epi16(src_2, coeff);                                         // etc.
				__m128i dst_3 = _mm_madd_epi16(src_3, coeff);
				__m128i dst_4 = _mm_madd_epi16(src_4, coeff);

				result_1 = _mm_add_epi32(result_1, dst_1);
				result_2 = _mm_add_epi32(result_2, dst_2);
				result_3 = _mm_add_epi32(result_3, dst_3);
				result_4 = _mm_add_epi32(result_4, dst_4);
			}

			if (notMod2) { // do last odd row
				__m128i src_p = load(reinterpret_cast<const __m128i*>(src_ptr + pitch_table[sizeMod2] + x));

				__m128i src_l = _mm_unpacklo_epi8(src_p, zero);
				__m128i src_h = _mm_unpackhi_epi8(src_p, zero);

				__m128i coeff = _mm_set1_epi16(current_coeff[sizeMod2]);

				__m128i dst_ll = _mm_mullo_epi16(src_l, coeff);   // Multiply by coefficient
				__m128i dst_lh = _mm_mulhi_epi16(src_l, coeff);
				__m128i dst_hl = _mm_mullo_epi16(src_h, coeff);
				__m128i dst_hh = _mm_mulhi_epi16(src_h, coeff);

				__m128i dst_1 = _mm_unpacklo_epi16(dst_ll, dst_lh); // Unpack to 32-bit integer
				__m128i dst_2 = _mm_unpackhi_epi16(dst_ll, dst_lh);
				__m128i dst_3 = _mm_unpacklo_epi16(dst_hl, dst_hh);
				__m128i dst_4 = _mm_unpackhi_epi16(dst_hl, dst_hh);

				result_1 = _mm_add_epi32(result_1, dst_1);
				result_2 = _mm_add_epi32(result_2, dst_2);
				result_3 = _mm_add_epi32(result_3, dst_3);
				result_4 = _mm_add_epi32(result_4, dst_4);
			}

			// Divide by 16348 (FPRound)
			result_1 = _mm_srai_epi32(result_1, 14);
			result_2 = _mm_srai_epi32(result_2, 14);
			result_3 = _mm_srai_epi32(result_3, 14);
			result_4 = _mm_srai_epi32(result_4, 14);

			// Pack and store
			__m128i result_l = _mm_packs_epi32(result_1, result_2);
			__m128i result_h = _mm_packs_epi32(result_3, result_4);
			__m128i result = _mm_packus_epi16(result_l, result_h);

			_mm_store_si128(reinterpret_cast<__m128i*>(dst + x), result);
		}

		// Leftover
		for (int x = wMod16; x < width; x++) {
			int result = 0;
			for (int i = 0; i < filter_size; i++) {
				result += (src_ptr + pitch_table[i])[x] * current_coeff[i];
			}
			result = ((result + 8192) / 16384);
			result = result > 255 ? 255 : result < 0 ? 0 : result;
			dst[x] = (BYTE)result;
		}

		dst += dst_pitch;
		current_coeff += filter_size;
	}
}


template<SSELoader load>
static void resize_v_ssse3_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  AVS_UNUSED(bits_per_pixel);
  AVS_UNUSED(storage);

	int filter_size = program->filter_size;
	short* current_coeff = program->pixel_coefficient;

	int wMod16 = (width / 16) * 16;

	__m128i zero = _mm_setzero_si128();
	__m128i coeff_unpacker = _mm_set_epi8(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0);

	for (int y = 0; y < target_height; y++) {
		int offset = program->pixel_offset[y];
		const BYTE* src_ptr = src + pitch_table[offset];

		for (int x = 0; x < wMod16; x += 16) {
			__m128i result_l = _mm_set1_epi16(32); // Init. with rounder ((1 << 6)/2 = 32)
			__m128i result_h = result_l;

			const BYTE* src2_ptr = src_ptr + x;

			for (int i = 0; i < filter_size; i++) {
				__m128i src_p = load(reinterpret_cast<const __m128i*>(src2_ptr));

				__m128i src_l = _mm_unpacklo_epi8(src_p, zero);
				__m128i src_h = _mm_unpackhi_epi8(src_p, zero);

				src_l = _mm_slli_epi16(src_l, 7);
				src_h = _mm_slli_epi16(src_h, 7);

				__m128i coeff = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(current_coeff + i));
				coeff = _mm_shuffle_epi8(coeff, coeff_unpacker);

				__m128i dst_l = _mm_mulhrs_epi16(src_l, coeff);   // Multiply by coefficient (SSSE3)
				__m128i dst_h = _mm_mulhrs_epi16(src_h, coeff);

				result_l = _mm_add_epi16(result_l, dst_l);
				result_h = _mm_add_epi16(result_h, dst_h);

				src2_ptr += src_pitch;
			}

			// Divide by 64
			result_l = _mm_srai_epi16(result_l, 6);
			result_h = _mm_srai_epi16(result_h, 6);

			// Pack and store
			__m128i result = _mm_packus_epi16(result_l, result_h);

			_mm_store_si128(reinterpret_cast<__m128i*>(dst + x), result);
		}

		// Leftover
		for (int x = wMod16; x < width; x++) {
			int result = 0;
			for (int i = 0; i < filter_size; i++) {
				result += (src_ptr + pitch_table[i])[x] * current_coeff[i];
			}
			result = ((result + 8192) / 16384);
			result = result > 255 ? 255 : result < 0 ? 0 : result;
			dst[x] = (BYTE)result;
		}

		dst += dst_pitch;
		current_coeff += filter_size;
	}
}

__forceinline static void resize_v_create_pitch_table(int* table, int pitch, int height) {
	table[0] = 0;
	for (int i = 1; i < height; i++) {
		table[i] = table[i - 1] + pitch;
	}
}



/***************************************
********* Horizontal Resizer** ********
***************************************/

static void resize_h_pointresize(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  AVS_UNUSED(bits_per_pixel);

	int wMod4 = width / 4 * 4;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < wMod4; x += 4) {
#define pixel(a) src[program->pixel_offset[x+a]]
			unsigned int data = (pixel(3) << 24) + (pixel(2) << 16) + (pixel(1) << 8) + pixel(0);
#undef pixel
			*((unsigned int *)(dst + x)) = data;
		}

		for (int x = wMod4; x < width; x++) {
			dst[x] = src[program->pixel_offset[x]];
		}

		dst += dst_pitch;
		src += src_pitch;
	}
}

template<typename pixel_t>
__global__ void kl_resize_h_pointresize(
	uint8_t* dst, const uint8_t* __restrict__ src, int dst_pitch, int src_pitch,
	const int* __restrict__ pixel_offset,
	int target_width, int target_height, float limit, int filter_size)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < target_width && y < target_height) {
		*(pixel_t*)(dst + x * sizeof(pixel_t) + y * dst_pitch) =
			*(const pixel_t*)(src + pixel_offset[x] * sizeof(pixel_t) + y * src_pitch);
	}
}

template<typename pixel_t>
void launch_resize_h_pointresize(
	BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch,
	const int* pixel_offset, const float* pixel_coefficient,
	int target_width, int target_height, float limit, int filter_size)
{
	dim3 threads(32, 16);
	dim3 blocks(nblocks(target_width, threads.x), nblocks(target_height, threads.y));

	kl_resize_h_pointresize<pixel_t> << <blocks, threads >> > (
		dst, src, dst_pitch, src_pitch,
		pixel_offset, target_width, target_height, limit, filter_size);
}

// make the resampling coefficient array mod8 or mod16 friendly for simd, padding non-used coeffs with zeros
static void resize_h_prepare_coeff_8or16(ResamplingProgram* p, IScriptEnvironment2* env, int alignFilterSize8or16) {
	p->filter_size_alignment = alignFilterSize8or16;
	int filter_size = AlignNumber(p->filter_size, alignFilterSize8or16);
	// for even non-simd it was aligned/padded as well, keep the same here
	int target_size = AlignNumber(p->target_size, ALIGN_RESIZER_TARGET_SIZE);

	// Copy existing coeff
	if (p->bits_per_pixel == 32) {
		float* new_coeff_float = (float*)env->Allocate(sizeof(float) * target_size * filter_size, 64, AVS_NORMAL_ALLOC);
		if (!new_coeff_float) {
			env->Free(new_coeff_float);
			env->ThrowError("Could not reserve memory in a resampler.");
		}
		std::fill_n(new_coeff_float, target_size * filter_size, 0.0f);
		float *dst_f = new_coeff_float, *src_f = p->pixel_coefficient_float;
		for (int i = 0; i < p->target_size; i++) {
			for (int j = 0; j < p->filter_size; j++) {
				dst_f[j] = src_f[j];
			}

			dst_f += filter_size;
			src_f += p->filter_size;
		}
		env->Free(p->pixel_coefficient_float);
		p->pixel_coefficient_float = new_coeff_float;
	}
	else {
		short* new_coeff = (short*)env->Allocate(sizeof(short) * target_size * filter_size, 64, AVS_NORMAL_ALLOC);
		if (!new_coeff) {
			env->Free(new_coeff);
			env->ThrowError("Could not reserve memory in a resampler.");
		}
		memset(new_coeff, 0, sizeof(short) * target_size * filter_size);
		short *dst = new_coeff, *src = p->pixel_coefficient;
		for (int i = 0; i < p->target_size; i++) {
			for (int j = 0; j < p->filter_size; j++) {
				dst[j] = src[j];
			}

			dst += filter_size;
			src += p->filter_size;

		}
		env->Free(p->pixel_coefficient);
		p->pixel_coefficient = new_coeff;
	}
}

template<typename pixel_t>
static void resize_h_c_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
	int filter_size = program->filter_size;

	typedef typename std::conditional < std::is_floating_point<pixel_t>::value, float, short>::type coeff_t;
	coeff_t *current_coeff;

	pixel_t limit = 0;
	if (!std::is_floating_point<pixel_t>::value) {  // floats are unscaled and uncapped
    if constexpr(sizeof(pixel_t) == 1) limit = 255;
    else if constexpr(sizeof(pixel_t) == 2) limit = pixel_t((1 << bits_per_pixel) - 1);
	}

	src_pitch = src_pitch / sizeof(pixel_t);
	dst_pitch = dst_pitch / sizeof(pixel_t);

	pixel_t* src0 = (pixel_t*)src;
	pixel_t* dst0 = (pixel_t*)dst;

	// external loop y is much faster
	for (int y = 0; y < height; y++) {
		if (!std::is_floating_point<pixel_t>::value)
			current_coeff = (coeff_t *)program->pixel_coefficient;
		else
			current_coeff = (coeff_t *)program->pixel_coefficient_float;
		for (int x = 0; x < width; x++) {
			int begin = program->pixel_offset[x];
			// todo: check whether int result is enough for 16 bit samples (can an int overflow because of 16384 scale or really need __int64?)
			typename std::conditional < sizeof(pixel_t) == 1, int, typename std::conditional < sizeof(pixel_t) == 2, __int64, float>::type >::type result;
			result = 0;
			for (int i = 0; i < filter_size; i++) {
				result += (src0 + y*src_pitch)[(begin + i)] * current_coeff[i];
			}
			if (!std::is_floating_point<pixel_t>::value) {  // floats are unscaled and uncapped
        if constexpr(sizeof(pixel_t) == 1)
					result = (result + (1 << (FPScale8bits - 1))) / (1 << FPScale8bits);
        else if constexpr(sizeof(pixel_t) == 2)
					result = (result + (1 << (FPScale16bits - 1))) / (1 << FPScale16bits);
				result = clamp(result, decltype(result)(0), decltype(result)(limit));
			}
			(dst0 + y*dst_pitch)[x] = (pixel_t)result;
			current_coeff += filter_size;
		}
	}
}

// vector primitives
// float2 += float2
static __device__ void operator+=(float2& a, float2 b) {
	a.x += b.x;
	a.y += b.y;
}
// float3 += float3
static __device__ void operator+=(float3& a, float3 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}
// float4 += float4
static __device__ void operator+=(float4& a, float4 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}
// float * uchar2
static __device__ float2 operator*(float a, uchar2 b) {
	float2 r = { a * b.x, a * b.y };
	return r;
}
// float * uchar3
static __device__ float3 operator*(float a, uchar3 b) {
	float3 r = { a * b.x, a * b.y, a * b.z };
	return r;
}
// float * uchar4
static __device__ float4 operator*(float a, uchar4 b) {
	float4 r = { a * b.x, a * b.y, a * b.z, a * b.w };
	return r;
}
// float * ushort3
static __device__ float3 operator*(float a, ushort3 b) {
	float3 r = { a * b.x, a * b.y, a * b.z };
	return r;
}
// float * ushort4
static __device__ float4 operator*(float a, ushort4 b) {
	float4 r = { a * b.x, a * b.y, a * b.z, a * b.w };
	return r;
}
// clamp(float2, int, int)
static __device__ float2 clamp(float2 a, float b, float c) {
	float2 r = { clamp<float>(a.x, b, c), clamp<float>(a.y, b, c) };
	return r;
}
// clamp(float3, int, int)
static __device__ float3 clamp(float3 a, float b, float c) {
	float3 r = { clamp<float>(a.x, b, c), clamp<float>(a.y, b, c), clamp<float>(a.z, b, c) };
	return r;
}
// clamp(float4, int, int)
static __device__ float4 clamp(float4 a, float b, float c) {
	float4 r = { clamp<float>(a.x, b, c), clamp<float>(a.y, b, c), clamp<float>(a.z, b, c), clamp<float>(a.w, b, c) };
	return r;
}
static __device__ uchar2 to_uchar2(float2 a) {
	uchar2 r = { (uint8_t)a.x, (uint8_t)a.y };
	return r;
}
static __device__ ushort2 to_ushort2(float2 a) {
	ushort2 r = { (uint16_t)a.x, (uint16_t)a.y };
	return r;
}
static __device__ uchar3 to_uchar3(float3 a) {
	uchar3 r = { (uint8_t)a.x, (uint8_t)a.y, (uint8_t)a.z };
	return r;
}
static __device__ uchar4 to_uchar4(float4 a) {
	uchar4 r = { (uint8_t)a.x, (uint8_t)a.y, (uint8_t)a.z, (uint8_t)a.w };
	return r;
}
static __device__ ushort3 to_ushort3(float3 a) {
	ushort3 r = { (uint16_t)a.x, (uint16_t)a.y, (uint16_t)a.z };
	return r;
}
static __device__ ushort4 to_ushort4(float4 a) {
	ushort4 r = { (uint16_t)a.x, (uint16_t)a.y, (uint16_t)a.z, (uint16_t)a.w };
	return r;
}
template <typename pixel_t> struct TypeHelper { };
template <> struct TypeHelper<uint8_t> {
	typedef float float_t;
	static __device__ uint8_t c(float a, float b, float c) { return (uint8_t)clamp<float>(a, b, c); };
};
template <> struct TypeHelper<uint16_t> {
	typedef float float_t;
	static __device__ uint16_t c(float a, float b, float c) { return (uint16_t)clamp<float>(a, b, c); };
};
template <> struct TypeHelper<float> {
	typedef float float_t;
	static __device__ float c(float a, float b, float c) { return a; };
};
template <> struct TypeHelper<uchar2> {
	typedef float2 float_t;
	static __device__ uchar2 c(float2 a, float b, float c) { return to_uchar2(clamp(a, b, c)); };
};
template <> struct TypeHelper<ushort2> {
	typedef float2 float_t;
	static __device__ ushort2 c(float2 a, float b, float c) { return to_ushort2(clamp(a, b, c)); };
};
template <> struct TypeHelper<float2> {
	typedef float2 float_t;
	static __device__ float2 c(float2 a, float b, float c) { return a; };
};
template <> struct TypeHelper<uchar3> {
	typedef float3 float_t;
	static __device__ uchar3 c(float3 a, float b, float c) { return to_uchar3(clamp(a, b, c)); };
};
template <> struct TypeHelper<uchar4> {
	typedef float4 float_t;
	static __device__ uchar4 c(float4 a, float b, float c) { return to_uchar4(clamp(a, b, c)); };
};
template <> struct TypeHelper<ushort3> {
	typedef float3 float_t;
	static __device__ ushort3 c(float3 a, float b, float c) { return to_ushort3(clamp(a, b, c)); };
};
template <> struct TypeHelper<ushort4> {
	typedef float4 float_t;
	static __device__ ushort4 c(float4 a, float b, float c) { return to_ushort4(clamp(a, b, c)); };
};
template <> struct TypeHelper<float3> {
	typedef float3 float_t;
	static __device__ float3 c(float3 a, float b, float c) { return a; };
};

template<typename pixel_t>
__global__ void kl_resize_h_planar(
	uint8_t* dst, const uint8_t* __restrict__ src, int dst_pitch, int src_pitch,
	const int* __restrict__ pixel_offset, const float* __restrict__ pixel_coefficient,
	int target_width, int target_height, float limit, int filter_size)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < target_width && y < target_height) {
		auto result = typename TypeHelper<pixel_t>::float_t();
		int x_offset = pixel_offset[x];
		for (int i = 0; i < filter_size; ++i) {
			auto sval = *(const pixel_t*)(src + (x_offset + i) * sizeof(pixel_t) + y * src_pitch);
			result += pixel_coefficient[i] * sval;
		}
		*(pixel_t*)(dst + x * sizeof(pixel_t) + y * dst_pitch) =
			TypeHelper<pixel_t>::c(result, 0.0f, limit);
	}
}

template<typename pixel_t>
void launch_resize_h_planar(
	BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch,
	const int* pixel_offset, const float* pixel_coefficient,
	int target_width, int target_height, float limit, int filter_size)
{
	dim3 threads(32, 16);
	dim3 blocks(nblocks(target_width, threads.x), nblocks(target_height, threads.y));

	kl_resize_h_planar<pixel_t> << <blocks, threads >> > (
		dst, src, dst_pitch, src_pitch,
		pixel_offset, pixel_coefficient, target_width, target_height, limit, filter_size);
}

//-------- 128 bit float Horizontals

__forceinline static void process_one_pixel_h_float(const float *src, int begin, int i, float *&current_coeff, __m128 &result) {
	// 2x4 pixels
	__m128 data_l_single = _mm_loadu_ps(reinterpret_cast<const float*>(src + begin + i * 8));
	__m128 data_h_single = _mm_loadu_ps(reinterpret_cast<const float*>(src + begin + i * 8 + 4));
	__m128 coeff_l = _mm_load_ps(reinterpret_cast<const float*>(current_coeff)); // always aligned
	__m128 coeff_h = _mm_load_ps(reinterpret_cast<const float*>(current_coeff + 4));
	__m128 dst_l = _mm_mul_ps(data_l_single, coeff_l); // Multiply by coefficient
	__m128 dst_h = _mm_mul_ps(data_h_single, coeff_h); // 4*(32bit*32bit=32bit)
	result = _mm_add_ps(result, dst_l); // accumulate result.
	result = _mm_add_ps(result, dst_h);
	current_coeff += 8;
}

template<int filtersizemod8>
__forceinline static void process_one_pixel_h_float_mask(const float *src, int begin, int i, float *&current_coeff, __m128 &result, __m128 &mask) {
	__m128 data_l_single;
	__m128 data_h_single;
	// 2x4 pixels
  if constexpr(filtersizemod8 > 4) { // keep low, mask high 4 pixels
		data_l_single = _mm_loadu_ps(reinterpret_cast<const float*>(src + begin + i * 8));
		data_h_single = _mm_loadu_ps(reinterpret_cast<const float*>(src + begin + i * 8 + 4));
		data_h_single = _mm_and_ps(data_h_single, mask);
	}
  else if constexpr(filtersizemod8 == 4) { // keep low, zero high 4 pixels
		data_l_single = _mm_loadu_ps(reinterpret_cast<const float*>(src + begin + i * 8));
		data_h_single = _mm_setzero_ps();
	}
	else { // filtersizemod8 1..3
		data_l_single = _mm_loadu_ps(reinterpret_cast<const float*>(src + begin + i * 8));
		data_l_single = _mm_and_ps(data_l_single, mask);
		data_h_single = _mm_setzero_ps();
	}
	__m128 coeff_l = _mm_load_ps(reinterpret_cast<const float*>(current_coeff)); // always aligned
	__m128 coeff_h = _mm_load_ps(reinterpret_cast<const float*>(current_coeff + 4));
	__m128 dst_l = _mm_mul_ps(data_l_single, coeff_l); // Multiply by coefficient
	__m128 dst_h = _mm_mul_ps(data_h_single, coeff_h); // 4*(32bit*32bit=32bit)
	result = _mm_add_ps(result, dst_l); // accumulate result.
	result = _mm_add_ps(result, dst_h);
	current_coeff += 8;
}

// filtersizealigned8: special: 1, 2. Generic: -1
template<int filtersizealigned8, int filtersizemod8>
static void resizer_h_ssse3_generic_float(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  AVS_UNUSED(bits_per_pixel);
	const int filter_size_numOfBlk8 = (filtersizealigned8 >= 1) ? filtersizealigned8 : (AlignNumber(program->filter_size, 8) / 8);

	const float *src = reinterpret_cast<const float *>(src8);
	float *dst = reinterpret_cast<float *>(dst8);
	dst_pitch /= sizeof(float);
	src_pitch /= sizeof(float);

	// OMG! 18.01.19
	// Protection against NaN
	// When reading the last 8 consecutive pixels from right side offsets, it would access beyond-last-pixel area.
	// One SIMD cycle reads 8 bytes from (src + begin + i * 8)
	// When program->filter_size mod 8 is 1..7 then some of the last pixels should be masked because there can be NaN garbage.
	// So it's not enough to mask the coefficients by zero. Theory: let's multiply offscreen elements by 0 which works for integer samples.
	// But we are using float, so since NaN * Zero is NaN which propagates further to NaN when hadd is summing up the pixel*coeff series

	__m128 mask;
	switch (filtersizemod8 & 3) {
	case 3: mask = _mm_castsi128_ps(_mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0)); break; // keep 0-1-2, drop #3
	case 2: mask = _mm_castsi128_ps(_mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0)); break; // keep 0-1, drop #2-3
	case 1: mask = _mm_castsi128_ps(_mm_setr_epi32(0xFFFFFFFF, 0, 0, 0)); break; // keep 0, drop #1-2-3
	default: break; // for mod4 = 0 no masking needed
	}

	const int pixels_per_cycle = 8; // doing 8 is faster than 4
	const int unsafe_limit = (program->overread_possible && filtersizemod8 != 0) ? (program->source_overread_beyond_targetx / pixels_per_cycle) * pixels_per_cycle : width;

	for (int y = 0; y < height; y++) {
		float* current_coeff = program->pixel_coefficient_float;

		// loop for clean, non-offscreen data
		for (int x = 0; x < unsafe_limit; x += pixels_per_cycle) {
			__m128 result1 = _mm_set1_ps(0.0f);
			__m128 result2 = result1;
			__m128 result3 = result1;
			__m128 result4 = result1;

			// 1-4
			int begin1 = program->pixel_offset[x + 0];
			int begin2 = program->pixel_offset[x + 1];
			int begin3 = program->pixel_offset[x + 2];
			int begin4 = program->pixel_offset[x + 3];

			// begin1, result1
			for (int i = 0; i < filter_size_numOfBlk8; i++)
				process_one_pixel_h_float(src, begin1, i, current_coeff, result1);

			// begin2, result2
			for (int i = 0; i < filter_size_numOfBlk8; i++)
				process_one_pixel_h_float(src, begin2, i, current_coeff, result2);

			// begin3, result3
			for (int i = 0; i < filter_size_numOfBlk8; i++)
				process_one_pixel_h_float(src, begin3, i, current_coeff, result3);

			// begin4, result4
			for (int i = 0; i < filter_size_numOfBlk8; i++)
				process_one_pixel_h_float(src, begin4, i, current_coeff, result4);

			// this part needs ssse3
			__m128 result12 = _mm_hadd_ps(result1, result2);
			__m128 result34 = _mm_hadd_ps(result3, result4);
			__m128 result = _mm_hadd_ps(result12, result34);

			_mm_stream_ps(reinterpret_cast<float*>(dst + x), result); // 4 results at a time

																																// 5-8
			result1 = _mm_set1_ps(0.0f);
			result2 = result1;
			result3 = result1;
			result4 = result1;

			begin1 = program->pixel_offset[x + 4];
			begin2 = program->pixel_offset[x + 5];
			begin3 = program->pixel_offset[x + 6];
			begin4 = program->pixel_offset[x + 7];

			// begin1, result1
			for (int i = 0; i < filter_size_numOfBlk8; i++)
				process_one_pixel_h_float(src, begin1, i, current_coeff, result1);

			// begin2, result2
			for (int i = 0; i < filter_size_numOfBlk8; i++)
				process_one_pixel_h_float(src, begin2, i, current_coeff, result2);

			// begin3, result3
			for (int i = 0; i < filter_size_numOfBlk8; i++)
				process_one_pixel_h_float(src, begin3, i, current_coeff, result3);

			// begin4, result4
			for (int i = 0; i < filter_size_numOfBlk8; i++)
				process_one_pixel_h_float(src, begin4, i, current_coeff, result4);

			// this part needs ssse3
			result12 = _mm_hadd_ps(result1, result2);
			result34 = _mm_hadd_ps(result3, result4);
			result = _mm_hadd_ps(result12, result34);

			_mm_stream_ps(reinterpret_cast<float*>(dst + x + 4), result); // 4 results at a time
		} // for x

			// possibly right-side offscreen
			// and the same for the rest with masking the last filtersize/8 chunk
		for (int x = unsafe_limit; x < width; x += 4) {
			__m128 result1 = _mm_set1_ps(0.0f);
			__m128 result2 = result1;
			__m128 result3 = result1;
			__m128 result4 = result1;

			int begin1 = program->pixel_offset[x + 0];
			int begin2 = program->pixel_offset[x + 1];
			int begin3 = program->pixel_offset[x + 2];
			int begin4 = program->pixel_offset[x + 3];

			// begin1, result1
			for (int i = 0; i < filter_size_numOfBlk8 - 1; i++)
				process_one_pixel_h_float(src, begin1, i, current_coeff, result1);
			if (begin1 < program->source_overread_offset)
				process_one_pixel_h_float(src, begin1, filter_size_numOfBlk8 - 1, current_coeff, result1);
			else
				process_one_pixel_h_float_mask<filtersizemod8>(src, begin1, filter_size_numOfBlk8 - 1, current_coeff, result1, mask);

			// begin2, result2
			for (int i = 0; i < filter_size_numOfBlk8 - 1; i++)
				process_one_pixel_h_float(src, begin2, i, current_coeff, result2);
			if (begin2 < program->source_overread_offset)
				process_one_pixel_h_float(src, begin2, filter_size_numOfBlk8 - 1, current_coeff, result2);
			else
				process_one_pixel_h_float_mask<filtersizemod8>(src, begin2, filter_size_numOfBlk8 - 1, current_coeff, result2, mask);

			// begin3, result3
			for (int i = 0; i < filter_size_numOfBlk8 - 1; i++)
				process_one_pixel_h_float(src, begin3, i, current_coeff, result3);
			if (begin3 < program->source_overread_offset)
				process_one_pixel_h_float(src, begin3, filter_size_numOfBlk8 - 1, current_coeff, result3);
			else
				process_one_pixel_h_float_mask<filtersizemod8>(src, begin3, filter_size_numOfBlk8 - 1, current_coeff, result3, mask);

			// begin4, result4
			for (int i = 0; i < filter_size_numOfBlk8 - 1; i++)
				process_one_pixel_h_float(src, begin4, i, current_coeff, result4);
			if (begin4 < program->source_overread_offset)
				process_one_pixel_h_float(src, begin4, filter_size_numOfBlk8 - 1, current_coeff, result4);
			else
				process_one_pixel_h_float_mask<filtersizemod8>(src, begin4, filter_size_numOfBlk8 - 1, current_coeff, result4, mask);

			// this part needs ssse3
			__m128 result12 = _mm_hadd_ps(result1, result2);
			__m128 result34 = _mm_hadd_ps(result3, result4);
			__m128 result = _mm_hadd_ps(result12, result34);

			_mm_stream_ps(reinterpret_cast<float*>(dst + x), result); // 4 results at a time

		} // for x

		dst += dst_pitch;
		src += src_pitch;
	}
	/*
	// check Nans
	dst -= dst_pitch * height;
	for (int y = 0; y < height; y++) {
	for (int x = 0; x < width; x += 4) {
	if (std::isnan(dst[x]))
	{
	x = x;
	}
	}
	dst += dst_pitch;
	}
	*/
}

//-------- 128 bit uint16_t Horizontals

template<bool lessthan16bit>
__forceinline static void process_one_pixel_h_uint16_t(const uint16_t *src, int begin, int i, short *&current_coeff, __m128i &result, const __m128i &shifttosigned) {
	__m128i data_single = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin + i * 8)); // 8 pixels
	if (!lessthan16bit)
		data_single = _mm_add_epi16(data_single, shifttosigned); // unsigned -> signed
	__m128i coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff)); // 8 coeffs
	result = _mm_add_epi32(result, _mm_madd_epi16(data_single, coeff));
	current_coeff += 8;
}

// filter_size <= 8 -> filter_size_align8 == 1 -> no loop, hope it'll be optimized
// filter_size <= 16 -> filter_size_align8 == 2 -> loop 0..1 hope it'll be optimized
// filter_size > 16 -> use parameter AlignNumber(program->filter_size_numOfFullBlk8, 8) / 8;
template<bool lessthan16bit, int filtersizealigned8, bool hasSSE41>
static void internal_resizer_h_sse34_generic_uint16_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
	// 1 and 2: special case for compiler optimization
	const int filter_size_numOfBlk8 = (filtersizealigned8 >= 1) ? filtersizealigned8 : (AlignNumber(program->filter_size, 8) / 8);

	const __m128i zero = _mm_setzero_si128();
	const __m128i shifttosigned = _mm_set1_epi16(-32768); // for 16 bits only
	const __m128i shiftfromsigned = _mm_set1_epi32(+32768 << FPScale16bits); // for 16 bits only
	const __m128i rounder = _mm_set_epi32(0, 0, 0, 1 << (FPScale16bits - 1)); // only once

	const uint16_t *src = reinterpret_cast<const uint16_t *>(src8);
	uint16_t *dst = reinterpret_cast<uint16_t *>(dst8);
	dst_pitch /= sizeof(uint16_t);
	src_pitch /= sizeof(uint16_t);

	__m128i clamp_limit = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1)); // clamp limit for <16 bits

	for (int y = 0; y < height; y++) {
		short* current_coeff = program->pixel_coefficient;

		for (int x = 0; x < width; x += 4) {
			__m128i result1 = rounder;
			__m128i result2 = result1;
			__m128i result3 = result1;
			__m128i result4 = result1;

			int begin1 = program->pixel_offset[x + 0];
			int begin2 = program->pixel_offset[x + 1];
			int begin3 = program->pixel_offset[x + 2];
			int begin4 = program->pixel_offset[x + 3];

			// this part is repeated 4 times
			// begin1, result1
			for (int i = 0; i < filter_size_numOfBlk8; i++)
				process_one_pixel_h_uint16_t<lessthan16bit>(src, begin1, i, current_coeff, result1, shifttosigned);

			// begin2, result2
			for (int i = 0; i < filter_size_numOfBlk8; i++)
				process_one_pixel_h_uint16_t<lessthan16bit>(src, begin2, i, current_coeff, result2, shifttosigned);

			// begin3, result3
			for (int i = 0; i < filter_size_numOfBlk8; i++)
				process_one_pixel_h_uint16_t<lessthan16bit>(src, begin3, i, current_coeff, result3, shifttosigned);

			// begin4, result4
			for (int i = 0; i < filter_size_numOfBlk8; i++)
				process_one_pixel_h_uint16_t<lessthan16bit>(src, begin4, i, current_coeff, result4, shifttosigned);

			const __m128i sumQuad12 = _mm_hadd_epi32(result1, result2); // L1L1L1L1 + L2L2L2L2 = L1L1 L2L2
			const __m128i sumQuad34 = _mm_hadd_epi32(result3, result4); // L3L3L3L3 + L4L4L4L4 = L3L3 L4L4
			__m128i result = _mm_hadd_epi32(sumQuad12, sumQuad34); // L1L1 L2L2 + L3L3 L4L4 = L1 L2 L3 L4

																														 // correct if signed, scale back, store
			if (!lessthan16bit)
				result = _mm_add_epi32(result, shiftfromsigned);
			result = _mm_srai_epi32(result, FPScale16bits); // shift back integer arithmetic 13 bits precision

			__m128i result_4x_uint16 = hasSSE41 ? _mm_packus_epi32(result, zero) : _MM_PACKUS_EPI32(result, zero); // 4*32+zeros = lower 4*16 OK
																																																						 // extra clamp for 10-14 bit
			if (lessthan16bit)
				result_4x_uint16 = hasSSE41 ? _mm_min_epu16(result_4x_uint16, clamp_limit) : _MM_MIN_EPU16(result_4x_uint16, clamp_limit);
			_mm_storel_epi64(reinterpret_cast<__m128i *>(dst + x), result_4x_uint16);

		}

		dst += dst_pitch;
		src += src_pitch;
	}
}

//-------- 128 bit uint16_t Horizontal Dispatcher

template<bool lessthan16bit, bool hasSSE41>
static void resizer_h_sse34_generic_uint16_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
	const int filter_size_numOfBlk8 = AlignNumber(program->filter_size, 8) / 8;

	if (filter_size_numOfBlk8 == 1)
		internal_resizer_h_sse34_generic_uint16_t<lessthan16bit, 1, hasSSE41>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
	else if (filter_size_numOfBlk8 == 2)
		internal_resizer_h_sse34_generic_uint16_t<lessthan16bit, 2, hasSSE41>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
	else // -1: basic method, use program->filter_size
		internal_resizer_h_sse34_generic_uint16_t<lessthan16bit, -1, hasSSE41>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
}

//-------- 128 bit uint16_t Verticals

template<bool lessthan16bit, int index>
__forceinline static void process_chunk_v_uint16_t(const uint16_t *src2_ptr, int src_pitch, __m128i &coeff01234567, __m128i &result_single_lo, __m128i &result_single_hi, const __m128i &shifttosigned) {
	// offset table generating is what preventing us from overaddressing
	// 0-1
	__m128i src_even = _mm_load_si128(reinterpret_cast<const __m128i*>(src2_ptr + index * src_pitch)); // 4x 16bit pixels
	__m128i src_odd = _mm_load_si128(reinterpret_cast<const __m128i*>(src2_ptr + (index + 1) * src_pitch));  // 4x 16bit pixels
	__m128i src_lo = _mm_unpacklo_epi16(src_even, src_odd);
	__m128i src_hi = _mm_unpackhi_epi16(src_even, src_odd);
	if (!lessthan16bit) {
		src_lo = _mm_add_epi16(src_lo, shifttosigned);
		src_hi = _mm_add_epi16(src_hi, shifttosigned);
	}
	__m128i coeff = _mm_shuffle_epi32(coeff01234567, ((index / 2) << 0) | ((index / 2) << 2) | ((index / 2) << 4) | ((index / 2) << 6)); // spread short pair
	result_single_lo = _mm_add_epi32(result_single_lo, _mm_madd_epi16(src_lo, coeff)); // a*b + c
	result_single_hi = _mm_add_epi32(result_single_hi, _mm_madd_epi16(src_hi, coeff)); // a*b + c
}

// program->filtersize: 1..16 special optimized, >8: normal
template<bool lessthan16bit, int _filter_size_numOfFullBlk8, int filtersizemod8, bool hasSSE41>
void internal_resize_v_sse_planar_uint16_t(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  AVS_UNUSED(storage);
	const int filter_size_numOfFullBlk8 = (_filter_size_numOfFullBlk8 >= 0) ? _filter_size_numOfFullBlk8 : (program->filter_size / 8);
	short* current_coeff = program->pixel_coefficient;

	// #define NON32_BYTES_ALIGNMENT
	// in AVS+ 32 bytes alignment is guaranteed
#ifdef NON32_BYTES_ALIGNMENT
	int wMod8 = (width / 8) * 8; // uint16: 8 at a time (2x128bit)
#endif

	const __m128i zero = _mm_setzero_si128();
	const __m128i shifttosigned = _mm_set1_epi16(-32768); // for 16 bits only
	const __m128i shiftfromsigned = _mm_set1_epi32(32768 << FPScale16bits); // for 16 bits only
	const __m128i rounder = _mm_set1_epi32(1 << (FPScale16bits - 1));

	const uint16_t* src = (uint16_t *)src0;
	uint16_t* dst = (uint16_t *)dst0;
	dst_pitch = dst_pitch / sizeof(uint16_t);
	src_pitch = src_pitch / sizeof(uint16_t);

	const int limit = (1 << bits_per_pixel) - 1;
	__m128i clamp_limit = _mm_set1_epi16((short)limit); // clamp limit for <16 bits

	for (int y = 0; y < target_height; y++) {
		int offset = program->pixel_offset[y];
		const uint16_t* src_ptr = src + pitch_table[offset] / sizeof(uint16_t);

#ifdef NON32_BYTES_ALIGNMENT
		for (int x = 0; x < wMod8; x += 8) { // 2x4 pixels at a time
#else
		for (int x = 0; x < width; x += 8) {
#endif
			__m128i result_single_lo = rounder;
			__m128i result_single_hi = rounder;

			const uint16_t* src2_ptr = src_ptr + x;

			for (int i = 0; i < filter_size_numOfFullBlk8; i++) {
				__m128i coeff01234567 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current_coeff + i * 8)); // 4x (2x16bit) shorts for even/odd

																																																					// offset table generating is what preventing us from overaddressing
																																																					// 0-1
				process_chunk_v_uint16_t<lessthan16bit, 0>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
				// 2-3
				process_chunk_v_uint16_t<lessthan16bit, 2>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
				// 4-5
				process_chunk_v_uint16_t<lessthan16bit, 4>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
				// 6-7
				process_chunk_v_uint16_t<lessthan16bit, 6>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
				src2_ptr += 8 * src_pitch;
			}

			// and the rest non-div8 chunk
			__m128i coeff01234567 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current_coeff + filter_size_numOfFullBlk8 * 8)); // 4x (2x16bit) shorts for even/odd
      if constexpr(filtersizemod8 >= 2)
				process_chunk_v_uint16_t<lessthan16bit, 0>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
      if constexpr(filtersizemod8 >= 4)
				process_chunk_v_uint16_t<lessthan16bit, 2>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
      if constexpr(filtersizemod8 >= 6)
				process_chunk_v_uint16_t<lessthan16bit, 4>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
      if constexpr(filtersizemod8 % 2) { // remaining odd one
				const int index = filtersizemod8 - 1;
				__m128i src_even = _mm_load_si128(reinterpret_cast<const __m128i*>(src2_ptr + index * src_pitch)); // 8x 16bit pixels
				if (!lessthan16bit)
					src_even = _mm_add_epi16(src_even, shifttosigned);
				__m128i coeff = _mm_shuffle_epi32(coeff01234567, ((index / 2) << 0) | ((index / 2) << 2) | ((index / 2) << 4) | ((index / 2) << 6));
				__m128i src_lo = _mm_unpacklo_epi16(src_even, zero); // insert zero after the unsigned->signed shift!
				__m128i src_hi = _mm_unpackhi_epi16(src_even, zero); // insert zero after the unsigned->signed shift!
				result_single_lo = _mm_add_epi32(result_single_lo, _mm_madd_epi16(src_lo, coeff)); // a*b + c
				result_single_hi = _mm_add_epi32(result_single_hi, _mm_madd_epi16(src_hi, coeff)); // a*b + c
			}

			// correct if signed, scale back, store
			__m128i result_lo = result_single_lo;
			__m128i result_hi = result_single_hi;
			if (!lessthan16bit) {
				result_lo = _mm_add_epi32(result_lo, shiftfromsigned);
				result_hi = _mm_add_epi32(result_hi, shiftfromsigned);
			}
			result_lo = _mm_srai_epi32(result_lo, FPScale16bits); // shift back integer arithmetic 13 bits precision
			result_hi = _mm_srai_epi32(result_hi, FPScale16bits);

			__m128i result_8x_uint16 = hasSSE41 ? _mm_packus_epi32(result_lo, result_hi) : _MM_PACKUS_EPI32(result_lo, result_hi);
			if (lessthan16bit)
				result_8x_uint16 = hasSSE41 ? _mm_min_epu16(result_8x_uint16, clamp_limit) : _MM_MIN_EPU16(result_8x_uint16, clamp_limit); // extra clamp for 10-14 bit
			_mm_store_si128(reinterpret_cast<__m128i *>(dst + x), result_8x_uint16);
		}

#ifdef NON32_BYTES_ALIGNMENT
		// Leftover, slow C
		for (int x = wMod8; x < width; x++) {
			int64_t result64 = 1 << (FPScale16bits - 1); // rounder
			const uint16_t* src2_ptr = src_ptr + x;
			for (int i = 0; i < program->filter_size; i++) {
				//result64 += (src_ptr + pitch_table[i] / sizeof(uint16_t))[x] * (int64_t)current_coeff[i];
				result64 += (int)(*src2_ptr) * (int64_t)current_coeff[i];
				src2_ptr += src_pitch;
			}
			int result = (int)(result64 / (1 << FPScale16bits)); // scale back 13 bits
			result = result > limit ? limit : result < 0 ? 0 : result; // clamp 10..16 bits
			dst[x] = (uint16_t)result;
		}
#endif

		dst += dst_pitch;
		current_coeff += program->filter_size;
		}
	}

//-------- uint16_t Vertical Dispatcher

template<bool lessthan16bit, bool hasSSE41>
void resize_v_sse_planar_uint16_t(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
	// template<bool lessthan16bit, int _filter_size_numOfFullBlk8, int filtersizemod8>
	// filtersize 1..16: to template for optimization
	switch (program->filter_size) {
	case 1:
		internal_resize_v_sse_planar_uint16_t<lessthan16bit, 0, 1, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	case 2:
		internal_resize_v_sse_planar_uint16_t<lessthan16bit, 0, 2, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	case 3:
		internal_resize_v_sse_planar_uint16_t<lessthan16bit, 0, 3, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	case 4:
		internal_resize_v_sse_planar_uint16_t<lessthan16bit, 0, 4, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	case 5:
		internal_resize_v_sse_planar_uint16_t<lessthan16bit, 0, 5, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	case 6:
		internal_resize_v_sse_planar_uint16_t<lessthan16bit, 0, 6, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	case 7:
		internal_resize_v_sse_planar_uint16_t<lessthan16bit, 0, 7, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	case 8:
		internal_resize_v_sse_planar_uint16_t<lessthan16bit, 1, 0, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	case 9:
		internal_resize_v_sse_planar_uint16_t<lessthan16bit, 1, 1, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	case 10:
		internal_resize_v_sse_planar_uint16_t<lessthan16bit, 1, 2, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	case 11:
		internal_resize_v_sse_planar_uint16_t<lessthan16bit, 1, 3, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	case 12:
		internal_resize_v_sse_planar_uint16_t<lessthan16bit, 1, 4, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	case 13:
		internal_resize_v_sse_planar_uint16_t<lessthan16bit, 1, 5, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	case 14:
		internal_resize_v_sse_planar_uint16_t<lessthan16bit, 1, 6, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	case 15:
		internal_resize_v_sse_planar_uint16_t<lessthan16bit, 1, 7, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	default:
		switch (program->filter_size & 7) {
		case 0:
			internal_resize_v_sse_planar_uint16_t<lessthan16bit, -1, 0, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
			break;
		case 1:
			internal_resize_v_sse_planar_uint16_t<lessthan16bit, -1, 1, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
			break;
		case 2:
			internal_resize_v_sse_planar_uint16_t<lessthan16bit, -1, 2, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
			break;
		case 3:
			internal_resize_v_sse_planar_uint16_t<lessthan16bit, -1, 3, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
			break;
		case 4:
			internal_resize_v_sse_planar_uint16_t<lessthan16bit, -1, 4, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
			break;
		case 5:
			internal_resize_v_sse_planar_uint16_t<lessthan16bit, -1, 5, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
			break;
		case 6:
			internal_resize_v_sse_planar_uint16_t<lessthan16bit, -1, 6, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
			break;
		case 7:
			internal_resize_v_sse_planar_uint16_t<lessthan16bit, -1, 7, hasSSE41>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
			break;
		}
		break;
	}
}

//-------- 128 bit float Verticals

template<int _filtersize>
static void internal_resize_v_sse2_planar_float(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  AVS_UNUSED(bits_per_pixel);
  AVS_UNUSED(storage);
	// 1..8: special case for compiler optimization
	const int filter_size = _filtersize >= 1 ? _filtersize : program->filter_size;
	float* current_coeff_float = program->pixel_coefficient_float;

	__m128i zero = _mm_setzero_si128();

	const float* src = (float *)src0;
	float* dst = (float *)dst0;
	dst_pitch = dst_pitch / sizeof(float);
	src_pitch = src_pitch / sizeof(float);

	const int fsmod4 = (filter_size / 4) * 4;
	for (int y = 0; y < target_height; y++) {
		int offset = program->pixel_offset[y];
		const float* src_ptr = src + pitch_table[offset] / sizeof(float);

		// 8 pixels/cycle (32 bytes)
		for (int x = 0; x < width; x += 8) {  // safe to process 8 floats, 32 bytes alignment is OK
			__m128 result_single_lo = _mm_set1_ps(0.0f);
			__m128 result_single_hi = _mm_set1_ps(0.0f);

			const float* src2_ptr = src_ptr + x;

			for (int i = 0; i < fsmod4; i += 4) {
				__m128 src_single_lo;
				__m128 src_single_hi;
				__m128 coeff0123 = _mm_loadu_ps(reinterpret_cast<const float*>(current_coeff_float + i)); // loads 4 floats
				__m128 coeff;

				// unroll 4x
				// #1
				src_single_lo = _mm_load_ps(reinterpret_cast<const float*>(src2_ptr + 0 * src_pitch));
				src_single_hi = _mm_load_ps(reinterpret_cast<const float*>(src2_ptr + 0 * src_pitch + 4));
				coeff = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(coeff0123), (0 << 0) | (0 << 2) | (0 << 4) | (0 << 6))); // spread 0th
				result_single_lo = _mm_add_ps(result_single_lo, _mm_mul_ps(coeff, src_single_lo));
				result_single_hi = _mm_add_ps(result_single_hi, _mm_mul_ps(coeff, src_single_hi));

				// #2
				src_single_lo = _mm_load_ps(reinterpret_cast<const float*>(src2_ptr + 1 * src_pitch));
				src_single_hi = _mm_load_ps(reinterpret_cast<const float*>(src2_ptr + 1 * src_pitch + 4));
				coeff = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(coeff0123), (1 << 0) | (1 << 2) | (1 << 4) | (1 << 6))); // spread 1st
				result_single_lo = _mm_add_ps(result_single_lo, _mm_mul_ps(coeff, src_single_lo));
				result_single_hi = _mm_add_ps(result_single_hi, _mm_mul_ps(coeff, src_single_hi));

				// #3
				src_single_lo = _mm_load_ps(reinterpret_cast<const float*>(src2_ptr + 2 * src_pitch));
				src_single_hi = _mm_load_ps(reinterpret_cast<const float*>(src2_ptr + 2 * src_pitch + 4));
				coeff = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(coeff0123), (2 << 0) | (2 << 2) | (2 << 4) | (2 << 6))); // spread 2nd
				result_single_lo = _mm_add_ps(result_single_lo, _mm_mul_ps(coeff, src_single_lo));
				result_single_hi = _mm_add_ps(result_single_hi, _mm_mul_ps(coeff, src_single_hi));

				// #4
				src_single_lo = _mm_load_ps(reinterpret_cast<const float*>(src2_ptr + 3 * src_pitch));
				src_single_hi = _mm_load_ps(reinterpret_cast<const float*>(src2_ptr + 3 * src_pitch + 4));
				coeff = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(coeff0123), (3 << 0) | (3 << 2) | (3 << 4) | (3 << 6))); // spread 3rd
				result_single_lo = _mm_add_ps(result_single_lo, _mm_mul_ps(coeff, src_single_lo));
				result_single_hi = _mm_add_ps(result_single_hi, _mm_mul_ps(coeff, src_single_hi));

				src2_ptr += 4 * src_pitch;
			}

			// one-by-one
			for (int i = fsmod4; i < filter_size; i++) {
				__m128 src_single_lo = _mm_load_ps(reinterpret_cast<const float*>(src2_ptr + 0 * src_pitch));
				__m128 src_single_hi = _mm_load_ps(reinterpret_cast<const float*>(src2_ptr + 0 * src_pitch + 4));
				__m128 coeff = _mm_load1_ps(reinterpret_cast<const float*>(current_coeff_float + i)); // loads 1, fills all 8 floats
				result_single_lo = _mm_add_ps(result_single_lo, _mm_mul_ps(coeff, src_single_lo)); // _mm_fmadd_ps(src_single, coeff, result_single); // a*b + c
				result_single_hi = _mm_add_ps(result_single_hi, _mm_mul_ps(coeff, src_single_hi)); // _mm_fmadd_ps(src_single, coeff, result_single); // a*b + c

				src2_ptr += src_pitch;
			}

			_mm_stream_ps(reinterpret_cast<float*>(dst + x), result_single_lo);
			_mm_stream_ps(reinterpret_cast<float*>(dst + x + 4), result_single_hi);
		}

#if 0
		// Leftover, Slow C
		for (int x = wMod4; x < width; x++) {
			float result = 0;
			const float* src2_ptr = src_ptr + x;
			for (int i = 0; i < filter_size; i++) {
				result += (*src2_ptr) * current_coeff_float[i];
				src2_ptr += src_pitch;
			}
			dst[x] = result;
		}
#endif
		dst += dst_pitch;
		current_coeff_float += filter_size;
	}
}

//-------- Float Vertical Dispatcher

void resize_v_sse2_planar_float(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
	// 1..8: special case for compiler optimization
	switch (program->filter_size) {
	case 1:
		internal_resize_v_sse2_planar_float<1>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	case 2:
		internal_resize_v_sse2_planar_float<2>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	case 3:
		internal_resize_v_sse2_planar_float<3>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	case 4:
		internal_resize_v_sse2_planar_float<4>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	case 5:
		internal_resize_v_sse2_planar_float<5>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	case 6:
		internal_resize_v_sse2_planar_float<6>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	case 7:
		internal_resize_v_sse2_planar_float<7>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	case 8:
		internal_resize_v_sse2_planar_float<8>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	default:
		internal_resize_v_sse2_planar_float<-1>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
		break;
	}
}


//-------- uint8_t Horizontal (8bit)

static void resizer_h_ssse3_generic(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  AVS_UNUSED(bits_per_pixel);

	int filter_size = AlignNumber(program->filter_size, 8) / 8;
	__m128i zero = _mm_setzero_si128();

	for (int y = 0; y < height; y++) {
		short* current_coeff = program->pixel_coefficient;
		for (int x = 0; x < width; x += 4) {
			__m128i result1 = _mm_setr_epi32(8192, 0, 0, 0);
			__m128i result2 = _mm_setr_epi32(8192, 0, 0, 0);
			__m128i result3 = _mm_setr_epi32(8192, 0, 0, 0);
			__m128i result4 = _mm_setr_epi32(8192, 0, 0, 0);

			int begin1 = program->pixel_offset[x + 0];
			int begin2 = program->pixel_offset[x + 1];
			int begin3 = program->pixel_offset[x + 2];
			int begin4 = program->pixel_offset[x + 3];

			for (int i = 0; i < filter_size; i++) {
				__m128i data, coeff, current_result;
				data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin1 + i * 8)); // 8 * 8 bit pixels
				data = _mm_unpacklo_epi8(data, zero); // make 8*16 bit pixels
				coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));  // 8 coeffs 14 bit scaled -> ushort OK
				current_result = _mm_madd_epi16(data, coeff);
				result1 = _mm_add_epi32(result1, current_result);

				current_coeff += 8;
			}

			for (int i = 0; i < filter_size; i++) {
				__m128i data, coeff, current_result;
				data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin2 + i * 8));
				data = _mm_unpacklo_epi8(data, zero);
				coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));
				current_result = _mm_madd_epi16(data, coeff);
				result2 = _mm_add_epi32(result2, current_result);

				current_coeff += 8;
			}

			for (int i = 0; i < filter_size; i++) {
				__m128i data, coeff, current_result;
				data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin3 + i * 8));
				data = _mm_unpacklo_epi8(data, zero);
				coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));
				current_result = _mm_madd_epi16(data, coeff);
				result3 = _mm_add_epi32(result3, current_result);

				current_coeff += 8;
			}

			for (int i = 0; i < filter_size; i++) {
				__m128i data, coeff, current_result;
				data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin4 + i * 8));
				data = _mm_unpacklo_epi8(data, zero);
				coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));
				current_result = _mm_madd_epi16(data, coeff);
				result4 = _mm_add_epi32(result4, current_result);

				current_coeff += 8;
			}

			__m128i result12 = _mm_hadd_epi32(result1, result2);
			__m128i result34 = _mm_hadd_epi32(result3, result4);
			__m128i result = _mm_hadd_epi32(result12, result34);

			result = _mm_srai_epi32(result, 14);

			result = _mm_packs_epi32(result, zero);
			result = _mm_packus_epi16(result, zero);

			*((int*)(dst + x)) = _mm_cvtsi128_si32(result);
		}

		dst += dst_pitch;
		src += src_pitch;
	}
}

static void resizer_h_ssse3_8(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  AVS_UNUSED(bits_per_pixel);

	__m128i zero = _mm_setzero_si128();

	for (int y = 0; y < height; y++) {
		short* current_coeff = program->pixel_coefficient;
		for (int x = 0; x < width; x += 4) {
			__m128i result1 = _mm_setr_epi32(8192, 0, 0, 0);
			__m128i result2 = _mm_setr_epi32(8192, 0, 0, 0);
			__m128i result3 = _mm_setr_epi32(8192, 0, 0, 0);
			__m128i result4 = _mm_setr_epi32(8192, 0, 0, 0);

			int begin1 = program->pixel_offset[x + 0];
			int begin2 = program->pixel_offset[x + 1];
			int begin3 = program->pixel_offset[x + 2];
			int begin4 = program->pixel_offset[x + 3];

			__m128i data, coeff, current_result;

			// Unroll 1
			data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin1));
			data = _mm_unpacklo_epi8(data, zero);
			coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));
			current_result = _mm_madd_epi16(data, coeff);
			result1 = _mm_add_epi32(result1, current_result);

			current_coeff += 8;

			// Unroll 2
			data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin2));
			data = _mm_unpacklo_epi8(data, zero);
			coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));
			current_result = _mm_madd_epi16(data, coeff);
			result2 = _mm_add_epi32(result2, current_result);

			current_coeff += 8;

			// Unroll 3
			data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin3));
			data = _mm_unpacklo_epi8(data, zero);
			coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));
			current_result = _mm_madd_epi16(data, coeff);
			result3 = _mm_add_epi32(result3, current_result);

			current_coeff += 8;

			// Unroll 4
			data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin4));
			data = _mm_unpacklo_epi8(data, zero);
			coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));
			current_result = _mm_madd_epi16(data, coeff);
			result4 = _mm_add_epi32(result4, current_result);

			current_coeff += 8;

			// Combine
			__m128i result12 = _mm_hadd_epi32(result1, result2);
			__m128i result34 = _mm_hadd_epi32(result3, result4);
			__m128i result = _mm_hadd_epi32(result12, result34);

			result = _mm_srai_epi32(result, 14);

			result = _mm_packs_epi32(result, zero);
			result = _mm_packus_epi16(result, zero);

			*((int*)(dst + x)) = _mm_cvtsi128_si32(result);
		}

		dst += dst_pitch;
		src += src_pitch;
	}
}

/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/

extern const FuncDefinition Resample_filters[] = {
	{ "PointResize",    BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f", FilteredResize::Create_PointResize },
	{ "BilinearResize", BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f", FilteredResize::Create_BilinearResize },
	{ "BicubicResize",  BUILTIN_FUNC_PREFIX, "cii[b]f[c]f[src_left]f[src_top]f[src_width]f[src_height]f", FilteredResize::Create_BicubicResize },
	{ "LanczosResize",  BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f[taps]i", FilteredResize::Create_LanczosResize },
	{ "Lanczos4Resize", BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f", FilteredResize::Create_Lanczos4Resize },
	{ "BlackmanResize", BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f[taps]i", FilteredResize::Create_BlackmanResize },
	{ "Spline16Resize", BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f", FilteredResize::Create_Spline16Resize },
	{ "Spline36Resize", BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f", FilteredResize::Create_Spline36Resize },
	{ "Spline64Resize", BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f", FilteredResize::Create_Spline64Resize },
	{ "GaussResize",    BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f[p]f", FilteredResize::Create_GaussianResize },
	{ "SincResize",     BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f[taps]i", FilteredResize::Create_SincResize },
	/**
	* Resize(PClip clip, dst_width, dst_height [src_left, src_top, src_width, int src_height,] )
	*
	* src_left et al.   =  when these optional arguments are given, the filter acts just like
	*                      a Crop was performed with those parameters before resizing, only faster
	**/

	{ 0 }
};


FilteredResizeH::FilteredResizeH(PClip _child, double subrange_left, double subrange_width,
	int target_width, ResamplingFunction* func, IScriptEnvironment* env_)
	: GenericVideoFilter(_child),
	resampling_program_luma(0), resampling_program_chroma(0),
	src_pitch_table_luma(0),
	src_pitch_luma(-1),
	filter_storage_luma(0), filter_storage_chroma(0)
{
	PNeoEnv env = env_;

	src_width = vi.width;
	src_height = vi.height;
	dst_width = target_width;
	dst_height = vi.height;

	pixelsize = vi.ComponentSize(); // AVS16
	bits_per_pixel = vi.BitsPerComponent();
	grey = vi.IsY();

	bool isRGBPfamily = vi.IsPlanarRGB() || vi.IsPlanarRGBA();

	if (target_width <= 0) {
		env->ThrowError("Resize: Width must be greater than 0.");
	}

	if (vi.IsPlanar() && !grey && !isRGBPfamily) {
		const int mask = (1 << vi.GetPlaneWidthSubsampling(PLANAR_U)) - 1;

		if (target_width & mask)
			env->ThrowError("Resize: Planar destination height must be a multiple of %d.", mask + 1);
	}

	auto env2 = static_cast<IScriptEnvironment2*>(env);

	// Main resampling program
	resampling_program_luma = func->GetResamplingProgram(vi.width, subrange_left, subrange_width, target_width, bits_per_pixel, env2);
	if (vi.IsPlanar() && !grey && !isRGBPfamily) {
		const int shift = vi.GetPlaneWidthSubsampling(PLANAR_U);
		const int shift_h = vi.GetPlaneHeightSubsampling(PLANAR_U);
		const int div = 1 << shift;


		resampling_program_chroma = func->GetResamplingProgram(
			vi.width >> shift,
			subrange_left / div,
			subrange_width / div,
			target_width >> shift,
			bits_per_pixel,
			env2);
	}

	// r2592+: no target_width mod4 check, (old avs needed for unaligned frames?)
	fast_resize = (env->GetCPUFlags() & CPUF_SSSE3) == CPUF_SSSE3 && vi.IsPlanar();

	if (false && resampling_program_luma->filter_size == 1 && vi.IsPlanar()) {
		// dead code?
		fast_resize = true;
		resampler_h_luma = resize_h_pointresize;
		resampler_h_chroma = resize_h_pointresize;
	}
	else if (!fast_resize) {

		// nonfast-resize: using V resizer for horizontal resizing between a turnleft/right

		// Create resampling program and pitch table
		src_pitch_table_luma = new int[vi.width];

		resampler_luma = FilteredResizeV::GetResampler(env->GetCPUFlags(), true, pixelsize, bits_per_pixel, filter_storage_luma, resampling_program_luma);
		if (vi.IsPlanar() && !grey && !isRGBPfamily) {
			resampler_chroma = FilteredResizeV::GetResampler(env->GetCPUFlags(), true, pixelsize, bits_per_pixel, filter_storage_chroma, resampling_program_chroma);
		}

		// Temporary buffer size
		temp_1_pitch = AlignNumber(vi.BytesFromPixels(src_height), FRAME_ALIGN);
		temp_2_pitch = AlignNumber(vi.BytesFromPixels(dst_height), FRAME_ALIGN);

		resize_v_create_pitch_table(src_pitch_table_luma, temp_1_pitch, src_width);

		// Initialize Turn function
		// see turn.cpp
		bool has_sse2 = (env->GetCPUFlags() & CPUF_SSE2) != 0;
		if (vi.IsRGB24()) {
			turn_left = turn_left_rgb24;
			turn_right = turn_right_rgb24;
		}
		else if (vi.IsRGB32()) {
			if (has_sse2) {
				turn_left = turn_left_rgb32_sse2;
				turn_right = turn_right_rgb32_sse2;
			}
			else {
				turn_left = turn_left_rgb32_c;
				turn_right = turn_right_rgb32_c;
			}
		}
		else if (vi.IsRGB48()) {
			turn_left = turn_left_rgb48_c;
			turn_right = turn_right_rgb48_c;
		}
		else if (vi.IsRGB64()) {
			if (has_sse2) {
				turn_left = turn_left_rgb64_sse2;
				turn_right = turn_right_rgb64_sse2;
			}
			else {
				turn_left = turn_left_rgb64_c;
				turn_right = turn_right_rgb64_c;
			}
		}
		else {
			switch (vi.ComponentSize()) {// AVS16
			case 1: // 8 bit
				if (has_sse2) {
					turn_left = turn_left_plane_8_sse2;
					turn_right = turn_right_plane_8_sse2;
				}
				else {
					turn_left = turn_left_plane_8_c;
					turn_right = turn_right_plane_8_c;
				}
				break;
			case 2: // 16 bit
				if (has_sse2) {
					turn_left = turn_left_plane_16_sse2;
					turn_right = turn_right_plane_16_sse2;
				}
				else {
					turn_left = turn_left_plane_16_c;
					turn_right = turn_right_plane_16_c;
				}
				break;
			default: // 32 bit
				if (has_sse2) {
					turn_left = turn_left_plane_32_sse2;
					turn_right = turn_right_plane_32_sse2;
				}
				else {
					turn_left = turn_left_plane_32_c;
					turn_right = turn_right_plane_32_c;
				}
			}
		}
	}
	else { // Planar + SSSE3 = use new horizontal resizer routines
		resampler_h_luma = GetResampler(env->GetCPUFlags(), true, pixelsize, bits_per_pixel, resampling_program_luma, env2);

		if (!grey && !isRGBPfamily) {
			resampler_h_chroma = GetResampler(env->GetCPUFlags(), true, pixelsize, bits_per_pixel, resampling_program_chroma, env2);
		}
	}

	// CUDA
	dev_program_luma.pixel_offset = std::unique_ptr<DeviceLocalData<int>>(
		new DeviceLocalData<int>(resampling_program_luma->pixel_offset, target_width, env));
	if (resampling_program_luma->pixel_coefficient_float) {
		dev_program_luma.pixel_coefficient = std::unique_ptr<DeviceLocalData<float>>(
			new DeviceLocalData<float>(resampling_program_luma->pixel_coefficient_float,
				resampling_program_luma->filter_size, env));
	}
	if (resampling_program_chroma) {
		dev_program_chroma.pixel_offset = std::unique_ptr<DeviceLocalData<int>>(
			new DeviceLocalData<int>(resampling_program_chroma->pixel_offset,
				target_width >> vi.GetPlaneWidthSubsampling(PLANAR_U), env));
		if (resampling_program_chroma->pixel_coefficient_float) {
			dev_program_chroma.pixel_coefficient = std::unique_ptr<DeviceLocalData<float>>(
				new DeviceLocalData<float>(resampling_program_chroma->pixel_coefficient_float,
					resampling_program_chroma->filter_size, env));
		}
	}

	if (resampling_program_luma->filter_size == 1) {
		if (vi.IsRGB() && !isRGBPfamily) {
			// packed RGB
			if (vi.IsRGB24())
				dev_resampler = launch_resize_h_pointresize<uchar3>;
			else if (vi.IsRGB32())
				dev_resampler = launch_resize_h_pointresize<uchar4>;
			else if (vi.IsRGB48())
				dev_resampler = launch_resize_h_pointresize<ushort3>;
			else if (vi.IsRGB64())
				dev_resampler = launch_resize_h_pointresize<ushort4>;
		}
		else if (vi.IsYUY2()) {
			// YUY2
			// if (pixelsize == 1)
			dev_resampler = launch_resize_h_pointresize<uchar2>;
		}
		else {
			// planar
			if (pixelsize == 1)
				dev_resampler = launch_resize_h_pointresize<uint8_t>;
			else if (pixelsize == 2)
				dev_resampler = launch_resize_h_pointresize<uint16_t>;
			else
				dev_resampler = launch_resize_h_pointresize<float>;
		}
	}
	else {
		if (vi.IsRGB() && !isRGBPfamily) {
			// packed RGB
			if (vi.IsRGB24())
				dev_resampler = launch_resize_h_planar<uchar3>;
			else if (vi.IsRGB32())
				dev_resampler = launch_resize_h_planar<uchar4>;
			else if (vi.IsRGB48())
				dev_resampler = launch_resize_h_planar<ushort3>;
			else if (vi.IsRGB64())
				dev_resampler = launch_resize_h_planar<ushort4>;
		}
		else if (vi.IsYUY2()) {
			// YUY2
			// if (pixelsize == 1)
			dev_resampler = launch_resize_h_planar<uchar2>;
		}
		else {
			// planar
			if (pixelsize == 1)
				dev_resampler = launch_resize_h_planar<uint8_t>;
			else if (pixelsize == 2)
				dev_resampler = launch_resize_h_planar<uint16_t>;
			else
				dev_resampler = launch_resize_h_planar<float>;
		}
	}

	// Change target video info size
	vi.width = target_width;
}

int __stdcall FilteredResizeH::SetCacheHints(int cachehints, int frame_range)
{
	if (cachehints == CACHE_GET_DEV_TYPE) {
		return GetDeviceTypes(child) &
			(DEV_TYPE_CPU | DEV_TYPE_CUDA);
	}
	return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
}

PVideoFrame __stdcall FilteredResizeH::GetFrame(int n, IScriptEnvironment* env_)
{
	PNeoEnv env = env_;
	PVideoFrame src = child->GetFrame(n, env);
	PVideoFrame dst = env->NewVideoFrame(vi);

	auto env2 = static_cast<IScriptEnvironment2*>(env);

	bool isRGBPfamily = vi.IsPlanarRGB() || vi.IsPlanarRGBA();

	if (IS_CUDA) {
		auto limit = (1 << bits_per_pixel) - 1;
		const int planesYUV[] = { 0/*PLANAR_Y*/, PLANAR_U, PLANAR_V, PLANAR_A };
		const int planesRGB[] = { 0/*PLANAR_G*/, PLANAR_B, PLANAR_R, PLANAR_A };
		const int* planes = vi.IsRGB() ? planesRGB : planesYUV;
		int numPlanes = vi.IsPlanar() ? vi.NumComponents() : 1;
		for (int p = 0; p < numPlanes; p++) {
			const int plane = planes[p];
			const auto program = (p > 0 && !vi.IsRGB()) ? resampling_program_chroma : resampling_program_luma;
			const auto dev_program = (p > 0 && !vi.IsRGB()) ? &dev_program_chroma : &dev_program_luma;
			const auto pixel_offset = dev_program->pixel_offset->GetData(env);
			const auto pixel_coefficient = dev_program->pixel_coefficient ? dev_program->pixel_coefficient->GetData(env) : nullptr;
			dev_resampler(dst->GetWritePtr(plane), src->GetReadPtr(plane), dst->GetPitch(plane), src->GetPitch(plane),
				pixel_offset, pixel_coefficient,
				dst->GetRowSize(plane) / vi.BytesFromPixels(1), dst->GetHeight(plane), (float)limit, program->filter_size);
			DEBUG_SYNC;
		}
	}
	else if (!fast_resize) {
		// e.g. not aligned, not mod4
		// temp_1_pitch and temp_2_pitch is pixelsize-aware
		BYTE* temp_1 = static_cast<BYTE*>(env2->Allocate(temp_1_pitch * src_width, FRAME_ALIGN, AVS_POOLED_ALLOC));
		BYTE* temp_2 = static_cast<BYTE*>(env2->Allocate(temp_2_pitch * dst_width, FRAME_ALIGN, AVS_POOLED_ALLOC));
		if (!temp_1 || !temp_2) {
			env2->Free(temp_1);
			env2->Free(temp_2);
			env->ThrowError("Could not reserve memory in a resampler.");
		}

		if (!vi.IsRGB() || isRGBPfamily) {
			// Y/G Plane
			turn_right(src->GetReadPtr(), temp_1, src_width * pixelsize, src_height, src->GetPitch(), temp_1_pitch); // * pixelsize: turn_right needs GetPlaneWidth full size
			resampler_luma(temp_2, temp_1, temp_2_pitch, temp_1_pitch, resampling_program_luma, src_height, dst_width, bits_per_pixel, src_pitch_table_luma, filter_storage_luma);
			turn_left(temp_2, dst->GetWritePtr(), dst_height * pixelsize, dst_width, temp_2_pitch, dst->GetPitch());

			if (isRGBPfamily)
			{
				turn_right(src->GetReadPtr(PLANAR_B), temp_1, src_width * pixelsize, src_height, src->GetPitch(PLANAR_B), temp_1_pitch); // * pixelsize: turn_right needs GetPlaneWidth full size
				resampler_luma(temp_2, temp_1, temp_2_pitch, temp_1_pitch, resampling_program_luma, src_height, dst_width, bits_per_pixel, src_pitch_table_luma, filter_storage_luma);
				turn_left(temp_2, dst->GetWritePtr(PLANAR_B), dst_height * pixelsize, dst_width, temp_2_pitch, dst->GetPitch(PLANAR_B));

				turn_right(src->GetReadPtr(PLANAR_R), temp_1, src_width * pixelsize, src_height, src->GetPitch(PLANAR_R), temp_1_pitch); // * pixelsize: turn_right needs GetPlaneWidth full size
				resampler_luma(temp_2, temp_1, temp_2_pitch, temp_1_pitch, resampling_program_luma, src_height, dst_width, bits_per_pixel, src_pitch_table_luma, filter_storage_luma);
				turn_left(temp_2, dst->GetWritePtr(PLANAR_R), dst_height * pixelsize, dst_width, temp_2_pitch, dst->GetPitch(PLANAR_R));
			}
			else if (!grey) {
				const int shift = vi.GetPlaneWidthSubsampling(PLANAR_U);
				const int shift_h = vi.GetPlaneHeightSubsampling(PLANAR_U);

				const int src_chroma_width = src_width >> shift;
				const int dst_chroma_width = dst_width >> shift;
				const int src_chroma_height = src_height >> shift_h;
				const int dst_chroma_height = dst_height >> shift_h;

				// turn_xxx: width * pixelsize: needs GetPlaneWidth-like full size
				// U Plane
				turn_right(src->GetReadPtr(PLANAR_U), temp_1, src_chroma_width * pixelsize, src_chroma_height, src->GetPitch(PLANAR_U), temp_1_pitch);
				resampler_luma(temp_2, temp_1, temp_2_pitch, temp_1_pitch, resampling_program_chroma, src_chroma_height, dst_chroma_width, bits_per_pixel, src_pitch_table_luma, filter_storage_chroma);
				turn_left(temp_2, dst->GetWritePtr(PLANAR_U), dst_chroma_height * pixelsize, dst_chroma_width, temp_2_pitch, dst->GetPitch(PLANAR_U));

				// V Plane
				turn_right(src->GetReadPtr(PLANAR_V), temp_1, src_chroma_width * pixelsize, src_chroma_height, src->GetPitch(PLANAR_V), temp_1_pitch);
				resampler_luma(temp_2, temp_1, temp_2_pitch, temp_1_pitch, resampling_program_chroma, src_chroma_height, dst_chroma_width, bits_per_pixel, src_pitch_table_luma, filter_storage_chroma);
				turn_left(temp_2, dst->GetWritePtr(PLANAR_V), dst_chroma_height * pixelsize, dst_chroma_width, temp_2_pitch, dst->GetPitch(PLANAR_V));
			}
			if (vi.IsYUVA() || vi.IsPlanarRGBA())
			{
				turn_right(src->GetReadPtr(PLANAR_A), temp_1, src_width * pixelsize, src_height, src->GetPitch(PLANAR_A), temp_1_pitch); // * pixelsize: turn_right needs GetPlaneWidth full size
				resampler_luma(temp_2, temp_1, temp_2_pitch, temp_1_pitch, resampling_program_luma, src_height, dst_width, bits_per_pixel, src_pitch_table_luma, filter_storage_luma);
				turn_left(temp_2, dst->GetWritePtr(PLANAR_A), dst_height * pixelsize, dst_width, temp_2_pitch, dst->GetPitch(PLANAR_A));
			}

		}
		else {
			// packed RGB
			// First left, then right. Reason: packed RGB bottom to top. Right+left shifts RGB24/RGB32 image to the opposite horizontal direction
			turn_left(src->GetReadPtr(), temp_1, vi.BytesFromPixels(src_width), src_height, src->GetPitch(), temp_1_pitch);
			resampler_luma(temp_2, temp_1, temp_2_pitch, temp_1_pitch, resampling_program_luma, vi.BytesFromPixels(src_height) / pixelsize, dst_width, bits_per_pixel, src_pitch_table_luma, filter_storage_luma);
			turn_right(temp_2, dst->GetWritePtr(), vi.BytesFromPixels(dst_height), dst_width, temp_2_pitch, dst->GetPitch());
		}

		env2->Free(temp_1);
		env2->Free(temp_2);
	}
	else {

		// Y Plane
		resampler_h_luma(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), resampling_program_luma, dst_width, dst_height, bits_per_pixel);

		if (isRGBPfamily) {
			resampler_h_luma(dst->GetWritePtr(PLANAR_B), src->GetReadPtr(PLANAR_B), dst->GetPitch(PLANAR_B), src->GetPitch(PLANAR_B), resampling_program_luma, dst_width, dst_height, bits_per_pixel);
			resampler_h_luma(dst->GetWritePtr(PLANAR_R), src->GetReadPtr(PLANAR_R), dst->GetPitch(PLANAR_R), src->GetPitch(PLANAR_R), resampling_program_luma, dst_width, dst_height, bits_per_pixel);
		}
		else if (!grey) {
			const int dst_chroma_width = dst_width >> vi.GetPlaneWidthSubsampling(PLANAR_U);
			const int dst_chroma_height = dst_height >> vi.GetPlaneHeightSubsampling(PLANAR_U);

			// U Plane
			resampler_h_chroma(dst->GetWritePtr(PLANAR_U), src->GetReadPtr(PLANAR_U), dst->GetPitch(PLANAR_U), src->GetPitch(PLANAR_U), resampling_program_chroma, dst_chroma_width, dst_chroma_height, bits_per_pixel);

			// V Plane
			resampler_h_chroma(dst->GetWritePtr(PLANAR_V), src->GetReadPtr(PLANAR_V), dst->GetPitch(PLANAR_V), src->GetPitch(PLANAR_V), resampling_program_chroma, dst_chroma_width, dst_chroma_height, bits_per_pixel);
		}
		if (vi.IsYUVA() || vi.IsPlanarRGBA())
		{
			resampler_h_luma(dst->GetWritePtr(PLANAR_A), src->GetReadPtr(PLANAR_A), dst->GetPitch(PLANAR_A), src->GetPitch(PLANAR_A), resampling_program_luma, dst_width, dst_height, bits_per_pixel);
		}

	}

	return dst;
}

ResamplerH FilteredResizeH::GetResampler(int CPU, bool aligned, int pixelsize, int bits_per_pixel, ResamplingProgram* program, IScriptEnvironment2* env)
{
  AVS_UNUSED(aligned);

	if (pixelsize == 1)
	{
		if (CPU & CPUF_SSSE3) {
			if (CPU & CPUF_AVX2) {
				// make the resampling coefficient array mod16 friendly for simd, padding non-used coeffs with zeros
				resize_h_prepare_coeff_8or16(program, env, 16);
				return resizer_h_avx2_generic_uint8_t;
			}
			else {
				// make the resampling coefficient array mod8 friendly for simd, padding non-used coeffs with zeros
				resize_h_prepare_coeff_8or16(program, env, 8);
				if (program->filter_size > 8)
					return resizer_h_ssse3_generic;
				else
					return resizer_h_ssse3_8; // no loop
			}
		}
		else { // C version
			return resize_h_c_planar<uint8_t>;
		}
	}
	else if (pixelsize == 2) {
		if (CPU & CPUF_SSSE3) {
			resize_h_prepare_coeff_8or16(program, env, 8); // alignment of 8 is enough for AVX2 uint16_t as well
			if (CPU & CPUF_AVX2) {
				if (bits_per_pixel < 16)
					return resizer_h_avx2_generic_uint16_t<true>;
				else
					return resizer_h_avx2_generic_uint16_t<false>;
			}
			else if (CPU & CPUF_SSE4_1) {
				if (bits_per_pixel < 16)
					return resizer_h_sse34_generic_uint16_t<true, true>;
				else
					return resizer_h_sse34_generic_uint16_t<false, true>;
			}
			else {
				// SSSE3 needed
				if (bits_per_pixel < 16)
					return resizer_h_sse34_generic_uint16_t<true, false>;
				else
					return resizer_h_sse34_generic_uint16_t<false, false>;
			}
    } else
			return resize_h_c_planar<uint16_t>;
  } else { //if (pixelsize == 4)

		if (CPU & CPUF_SSSE3) {
			resize_h_prepare_coeff_8or16(program, env, ALIGN_FLOAT_RESIZER_COEFF_SIZE); // alignment of 8 is enough for AVX2 float as well

			const int filtersizealign8 = AlignNumber(program->filter_size, 8);
			const int filtersizemod8 = program->filter_size & 7;

			if (CPU & CPUF_AVX2) {
				if (filtersizealign8 == 8) {
					switch (filtersizemod8) {
					case 0: return resizer_h_avx2_generic_float<1, 0>;
					case 1: return resizer_h_avx2_generic_float<1, 1>;
					case 2: return resizer_h_avx2_generic_float<1, 2>;
					case 3: return resizer_h_avx2_generic_float<1, 3>;
					case 4: return resizer_h_avx2_generic_float<1, 4>;
					case 5: return resizer_h_avx2_generic_float<1, 5>;
					case 6: return resizer_h_avx2_generic_float<1, 6>;
					case 7: return resizer_h_avx2_generic_float<1, 7>;
					}
				}
				else if (filtersizealign8 == 16) {
					switch (filtersizemod8) {
					case 0: return resizer_h_avx2_generic_float<2, 0>;
					case 1: return resizer_h_avx2_generic_float<2, 1>;
					case 2: return resizer_h_avx2_generic_float<2, 2>;
					case 3: return resizer_h_avx2_generic_float<2, 3>;
					case 4: return resizer_h_avx2_generic_float<2, 4>;
					case 5: return resizer_h_avx2_generic_float<2, 5>;
					case 6: return resizer_h_avx2_generic_float<2, 6>;
					case 7: return resizer_h_avx2_generic_float<2, 7>;
					}
				}
				else {
					switch (filtersizemod8) {
					case 0: return resizer_h_avx2_generic_float<-1, 0>;
					case 1: return resizer_h_avx2_generic_float<-1, 1>;
					case 2: return resizer_h_avx2_generic_float<-1, 2>;
					case 3: return resizer_h_avx2_generic_float<-1, 3>;
					case 4: return resizer_h_avx2_generic_float<-1, 4>;
					case 5: return resizer_h_avx2_generic_float<-1, 5>;
					case 6: return resizer_h_avx2_generic_float<-1, 6>;
					case 7: return resizer_h_avx2_generic_float<-1, 7>;
					}
				}
			}
			// SSSE3
			if (filtersizealign8 == 8) {
				switch (filtersizemod8) {
				case 0: return resizer_h_ssse3_generic_float<1, 0>;
				case 1: return resizer_h_ssse3_generic_float<1, 1>;
				case 2: return resizer_h_ssse3_generic_float<1, 2>;
				case 3: return resizer_h_ssse3_generic_float<1, 3>;
				case 4: return resizer_h_ssse3_generic_float<1, 4>;
				case 5: return resizer_h_ssse3_generic_float<1, 5>;
				case 6: return resizer_h_ssse3_generic_float<1, 6>;
				case 7: return resizer_h_ssse3_generic_float<1, 7>;
				}
			}
			else if (filtersizealign8 == 16) {
				switch (filtersizemod8) {
				case 0: return resizer_h_ssse3_generic_float<2, 0>;
				case 1: return resizer_h_ssse3_generic_float<2, 1>;
				case 2: return resizer_h_ssse3_generic_float<2, 2>;
				case 3: return resizer_h_ssse3_generic_float<2, 3>;
				case 4: return resizer_h_ssse3_generic_float<2, 4>;
				case 5: return resizer_h_ssse3_generic_float<2, 5>;
				case 6: return resizer_h_ssse3_generic_float<2, 6>;
				case 7: return resizer_h_ssse3_generic_float<2, 7>;
				}
			}
			else {
				switch (filtersizemod8) {
				case 0: return resizer_h_ssse3_generic_float<-1, 0>;
				case 1: return resizer_h_ssse3_generic_float<-1, 1>;
				case 2: return resizer_h_ssse3_generic_float<-1, 2>;
				case 3: return resizer_h_ssse3_generic_float<-1, 3>;
				case 4: return resizer_h_ssse3_generic_float<-1, 4>;
				case 5: return resizer_h_ssse3_generic_float<-1, 5>;
				case 6: return resizer_h_ssse3_generic_float<-1, 6>;
				case 7: return resizer_h_ssse3_generic_float<-1, 7>;
				}
			}
		}
		return resize_h_c_planar<float>;
	}
}

FilteredResizeH::~FilteredResizeH(void)
{
	if (resampling_program_luma) { delete resampling_program_luma; }
	if (resampling_program_chroma) { delete resampling_program_chroma; }
	if (src_pitch_table_luma) { delete[] src_pitch_table_luma; }
}

/***************************************
***** Filtered Resize - Vertical ******
***************************************/

FilteredResizeV::FilteredResizeV(PClip _child, double subrange_top, double subrange_height,
	int target_height, ResamplingFunction* func, IScriptEnvironment* env)
	: GenericVideoFilter(_child),
	resampling_program_luma(0), resampling_program_chroma(0),
	filter_storage_luma_aligned(0),
	filter_storage_chroma_aligned(0)
{
	if (target_height <= 0)
		env->ThrowError("Resize: Height must be greater than 0.");

	pixelsize = vi.ComponentSize(); // AVS16
	bits_per_pixel = vi.BitsPerComponent();
	grey = vi.IsY();
	bool isRGBPfamily = vi.IsPlanarRGB() || vi.IsPlanarRGBA();

	if (vi.IsPlanar() && !grey && !isRGBPfamily) {
		const int mask = (1 << vi.GetPlaneHeightSubsampling(PLANAR_U)) - 1;

		if (target_height & mask)
			env->ThrowError("Resize: Planar destination height must be a multiple of %d.", mask + 1);
	}

	auto env2 = static_cast<IScriptEnvironment2*>(env);

	if (vi.IsRGB() && !isRGBPfamily)
		subrange_top = vi.height - subrange_top - subrange_height; // packed RGB upside down


																															 // Create resampling program and pitch table
	resampling_program_luma = func->GetResamplingProgram(vi.height, subrange_top, subrange_height, target_height, bits_per_pixel, env2);
	resampler_luma_aligned = GetResampler(env->GetCPUFlags(), true, pixelsize, bits_per_pixel, filter_storage_luma_aligned, resampling_program_luma);

	if (vi.IsPlanar() && !grey && !isRGBPfamily) {
		const int shift = vi.GetPlaneHeightSubsampling(PLANAR_U);
		const int div = 1 << shift;

		resampling_program_chroma = func->GetResamplingProgram(
			vi.height >> shift,
			subrange_top / div,
			subrange_height / div,
			target_height >> shift,
			bits_per_pixel,
			env2);

		resampler_chroma_aligned = GetResampler(env->GetCPUFlags(), true, pixelsize, bits_per_pixel, filter_storage_chroma_aligned, resampling_program_chroma);
	}

	// CUDA
	dev_program_luma.pixel_offset = std::unique_ptr<DeviceLocalData<int>>(
		new DeviceLocalData<int>(resampling_program_luma->pixel_offset, target_height, env));
	if (resampling_program_luma->pixel_coefficient_float) {
		dev_program_luma.pixel_coefficient = std::unique_ptr<DeviceLocalData<float>>(
			new DeviceLocalData<float>(resampling_program_luma->pixel_coefficient_float,
				resampling_program_luma->filter_size, env));
	}
	if (resampling_program_chroma) {
		dev_program_chroma.pixel_offset = std::unique_ptr<DeviceLocalData<int>>(
			new DeviceLocalData<int>(resampling_program_chroma->pixel_offset, 
				target_height >> vi.GetPlaneHeightSubsampling(PLANAR_U), env));
		if (resampling_program_chroma->pixel_coefficient_float) {
			dev_program_chroma.pixel_coefficient = std::unique_ptr<DeviceLocalData<float>>(
				new DeviceLocalData<float>(resampling_program_chroma->pixel_coefficient_float,
					resampling_program_chroma->filter_size, env));
		}
	}

	if (resampling_program_luma->filter_size == 1) {
		if (pixelsize == 1)
			dev_resampler = launch_resize_v_planar_pointresize<uint8_t>;
		else if (pixelsize == 2)
			dev_resampler = launch_resize_v_planar_pointresize<uint16_t>;
		else
			dev_resampler = launch_resize_v_planar_pointresize<float>;
	}
	else {
		if (pixelsize == 1)
			dev_resampler = launch_resize_v_planar<uint8_t>;
		else if (pixelsize == 2)
			dev_resampler = launch_resize_v_planar<uint16_t>;
		else
			dev_resampler = launch_resize_v_planar<float>;
	}

	// Change target video info size
	vi.height = target_height;
}

int __stdcall FilteredResizeV::SetCacheHints(int cachehints, int frame_range)
{
	if (cachehints == CACHE_GET_DEV_TYPE) {
		return GetDeviceTypes(child) &
			(DEV_TYPE_CPU | DEV_TYPE_CUDA);
	}
	return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
}

PVideoFrame __stdcall FilteredResizeV::GetFrame(int n, IScriptEnvironment* env_)
{
	PNeoEnv env = env_;
	PVideoFrame src = child->GetFrame(n, env);
	PVideoFrame dst = env->NewVideoFrame(vi);
	int src_pitch = src->GetPitch();
	int dst_pitch = dst->GetPitch();
	const BYTE* srcp = src->GetReadPtr();
	BYTE* dstp = dst->GetWritePtr();

	bool isRGBPfamily = vi.IsPlanarRGB() || vi.IsPlanarRGBA();

	if (IS_CUDA) {
		auto limit = (1 << bits_per_pixel) - 1;
		const int planesYUV[] = { 0/*PLANAR_Y*/, PLANAR_U, PLANAR_V, PLANAR_A };
		const int planesRGB[] = { 0/*PLANAR_G*/, PLANAR_B, PLANAR_R, PLANAR_A };
		const int* planes = isRGBPfamily ? planesRGB : planesYUV;
		int numPlanes = vi.IsPlanar() ? vi.NumComponents() : 1;
		for (int p = 0; p < numPlanes; p++) {
			const int plane = planes[p];
			const auto program = (p > 0 && !isRGBPfamily) ? resampling_program_chroma : resampling_program_luma;
			const auto dev_program = (p > 0 && !isRGBPfamily) ? &dev_program_chroma : &dev_program_luma;
			const auto pixel_offset = dev_program->pixel_offset->GetData(env);
			const auto pixel_coefficient = dev_program->pixel_coefficient ? dev_program->pixel_coefficient->GetData(env) : nullptr;
			dev_resampler(dst->GetWritePtr(plane), src->GetReadPtr(plane), dst->GetPitch(plane), src->GetPitch(plane),
				pixel_offset, pixel_coefficient, dst->GetRowSize(plane) / pixelsize, dst->GetHeight(plane), (float)limit, program->filter_size);
			DEBUG_SYNC;
		}
	}
	else {
		// Create pitch table
		int* src_pitch_table_luma = static_cast<int*>(env->Allocate(sizeof(int) * src->GetHeight(), 32, AVS_POOLED_ALLOC));
		if (!src_pitch_table_luma) {
			env->ThrowError("Could not reserve memory in a resampler.");
		}

		resize_v_create_pitch_table(src_pitch_table_luma, src->GetPitch(), src->GetHeight());

		int* src_pitch_table_chromaU = NULL;
		int* src_pitch_table_chromaV = NULL;
		if ((!grey && vi.IsPlanar() && !isRGBPfamily)) {
			src_pitch_table_chromaU = static_cast<int*>(env->Allocate(sizeof(int) * src->GetHeight(PLANAR_U), 32, AVS_POOLED_ALLOC));
			src_pitch_table_chromaV = static_cast<int*>(env->Allocate(sizeof(int) * src->GetHeight(PLANAR_V), 32, AVS_POOLED_ALLOC));
			if (!src_pitch_table_chromaU || !src_pitch_table_chromaV) {
				env->Free(src_pitch_table_chromaU);
				env->Free(src_pitch_table_chromaV);
				env->ThrowError("Could not reserve memory in a resampler.");
			}

			resize_v_create_pitch_table(src_pitch_table_chromaU, src->GetPitch(PLANAR_U), src->GetHeight(PLANAR_U));
			resize_v_create_pitch_table(src_pitch_table_chromaV, src->GetPitch(PLANAR_V), src->GetHeight(PLANAR_V));
		}

		// Do resizing
		int work_width = vi.IsPlanar() ? vi.width : vi.BytesFromPixels(vi.width) / pixelsize; // packed RGB: or vi.width * vi.NumComponent()
																																													// alignment to FRAME_ALIGN is guaranteed
		resampler_luma_aligned(dstp, srcp, dst_pitch, src_pitch, resampling_program_luma, work_width, vi.height, bits_per_pixel, src_pitch_table_luma, filter_storage_luma_aligned);
		if (isRGBPfamily)
		{
			src_pitch = src->GetPitch(PLANAR_B);
			dst_pitch = dst->GetPitch(PLANAR_B);
			srcp = src->GetReadPtr(PLANAR_B);
			dstp = dst->GetWritePtr(PLANAR_B);
			// alignment to FRAME_ALIGN is guaranteed
			resampler_luma_aligned(dstp, srcp, dst_pitch, src_pitch, resampling_program_luma, work_width, vi.height, bits_per_pixel, src_pitch_table_luma, filter_storage_luma_aligned);
			src_pitch = src->GetPitch(PLANAR_R);
			dst_pitch = dst->GetPitch(PLANAR_R);
			srcp = src->GetReadPtr(PLANAR_R);
			dstp = dst->GetWritePtr(PLANAR_R);
			// alignment to FRAME_ALIGN is guaranteed
			resampler_luma_aligned(dstp, srcp, dst_pitch, src_pitch, resampling_program_luma, work_width, vi.height, bits_per_pixel, src_pitch_table_luma, filter_storage_luma_aligned);
		}
		else if (!grey && vi.IsPlanar()) {
			int width = vi.width >> vi.GetPlaneWidthSubsampling(PLANAR_U);
			int height = vi.height >> vi.GetPlaneHeightSubsampling(PLANAR_U);

			// Plane U resizing
			src_pitch = src->GetPitch(PLANAR_U);
			dst_pitch = dst->GetPitch(PLANAR_U);
			srcp = src->GetReadPtr(PLANAR_U);
			dstp = dst->GetWritePtr(PLANAR_U);

			// alignment to FRAME_ALIGN is guaranteed
			resampler_chroma_aligned(dstp, srcp, dst_pitch, src_pitch, resampling_program_chroma, width, height, bits_per_pixel, src_pitch_table_chromaU, filter_storage_chroma_aligned);

			// Plane V resizing
			src_pitch = src->GetPitch(PLANAR_V);
			dst_pitch = dst->GetPitch(PLANAR_V);
			srcp = src->GetReadPtr(PLANAR_V);
			dstp = dst->GetWritePtr(PLANAR_V);

			// alignment to FRAME_ALIGN is guaranteed
			resampler_chroma_aligned(dstp, srcp, dst_pitch, src_pitch, resampling_program_chroma, width, height, bits_per_pixel, src_pitch_table_chromaV, filter_storage_chroma_aligned);
		}

		// Free pitch table
		env->Free(src_pitch_table_luma);
		env->Free(src_pitch_table_chromaU);
		env->Free(src_pitch_table_chromaV);
	}

	return dst;
}

ResamplerV FilteredResizeV::GetResampler(int CPU, bool aligned, int pixelsize, int bits_per_pixel, void*& storage, ResamplingProgram* program)
{
  AVS_UNUSED(storage);
	if (program->filter_size == 1) {
		// Fast pointresize
		switch (pixelsize) // AVS16
		{
		case 1: return resize_v_planar_pointresize<uint8_t>;
		case 2: return resize_v_planar_pointresize<uint16_t>;
		default: // case 4:
			return resize_v_planar_pointresize<float>;
		}
	}
	else {
		// Other resizers
		if (pixelsize == 1)
		{
			if (CPU & CPUF_SSSE3) {
				if (aligned && (CPU & CPUF_AVX2)) {
					return resize_v_avx2_planar_uint8_t;
				}
				if (aligned && (CPU & CPUF_SSE4_1)) {
					return resize_v_ssse3_planar<simd_load_streaming>;
				}
				else if (aligned) { // SSSE3 aligned
					return resize_v_ssse3_planar<simd_load_aligned>;
				}
				else if (CPU & CPUF_SSE3) { // SSE3 lddqu
					return resize_v_ssse3_planar<simd_load_unaligned_sse3>;
				}
				else { // SSSE3 unaligned
					return resize_v_ssse3_planar<simd_load_unaligned>;
				}
			}
			else if (CPU & CPUF_SSE2) {
				if (aligned && CPU & CPUF_SSE4_1) { // SSE4.1 movntdqa constantly provide ~2% performance increase in my testing
					return resize_v_sse2_planar<simd_load_streaming>;
				}
				else if (aligned) { // SSE2 aligned
					return resize_v_sse2_planar<simd_load_aligned>;
				}
				else if (CPU & CPUF_SSE3) { // SSE2 lddqu
					return resize_v_sse2_planar<simd_load_unaligned_sse3>;
				}
				else { // SSE2 unaligned
					return resize_v_sse2_planar<simd_load_unaligned>;
				}
#ifdef X86_32
			}
			else if (CPU & CPUF_MMX) {
				return resize_v_mmx_planar;
#endif
			}
			else { // C version
				return resize_v_c_planar<uint8_t>;
			}
		}
		else if (pixelsize == 2) {
			if (aligned && (CPU & CPUF_AVX2)) {
				if (bits_per_pixel<16)
					return resize_v_avx2_planar_uint16_t<true>;
				else
					return resize_v_avx2_planar_uint16_t<false>;
			}
			else if (aligned && (CPU & CPUF_SSE4_1)) {
				if (bits_per_pixel < 16)
					return resize_v_sse_planar_uint16_t<true, true>;
				else
					return resize_v_sse_planar_uint16_t<false, true>;
			}
			else if (aligned && (CPU & CPUF_SSE2)) {
				if (bits_per_pixel < 16)
					return resize_v_sse_planar_uint16_t<true, false>;
				else
					return resize_v_sse_planar_uint16_t<false, false>;
			}
			else { // C version
				return resize_v_c_planar<uint16_t>;
			}
		}
		else { // pixelsize== 4
			if (aligned && (CPU & CPUF_AVX2)) {
				return resize_v_avx2_planar_float;
			}
			else if (aligned && (CPU & CPUF_SSE2)) {
				return resize_v_sse2_planar_float;
      } else {
				return resize_v_c_planar<float>;
			}
		}
	}
}

FilteredResizeV::~FilteredResizeV(void)
{
	if (resampling_program_luma) { delete resampling_program_luma; }
	if (resampling_program_chroma) { delete resampling_program_chroma; }
}


/**********************************************
*******   Resampling Factory Methods   *******
**********************************************/

PClip FilteredResize::CreateResizeH(PClip clip, double subrange_left, double subrange_width, int target_width,
	ResamplingFunction* func, IScriptEnvironment* env)
{
	const VideoInfo& vi = clip->GetVideoInfo();
	if (subrange_left == 0 && subrange_width == target_width && subrange_width == vi.width) {
		return clip;
	}

	if (subrange_left == int(subrange_left) && subrange_width == target_width
		&& subrange_left >= 0 && subrange_left + subrange_width <= vi.width) {
		const int mask = ((vi.IsYUV() || vi.IsYUVA()) && !vi.IsY()) ? (1 << vi.GetPlaneWidthSubsampling(PLANAR_U)) - 1 : 0;

		if (((int(subrange_left) | int(subrange_width)) & mask) == 0) {
			//return new Crop(int(subrange_left), 0, int(subrange_width), vi.height, 0, clip, env);
			AVSValue args[] = { clip, int(subrange_left), 0, int(subrange_width), vi.height, false };
			return env->Invoke("Crop", AVSValue(args, 6)).AsClip();
		}
	}

	// Convert interleaved yuv to planar yuv
	PClip result = clip;
	if (vi.IsYUY2()) {
		// result = new ConvertYUY2ToYV16(result, env);
		AVSValue arg = result;
		result = env->Invoke("ConvertToYV16", AVSValue(&arg, 1)).AsClip();
	}
	result = new FilteredResizeH(result, subrange_left, subrange_width, target_width, func, env);
	if (vi.IsYUY2()) {
		//result = new ConvertYV16ToYUY2(result,  env);
		AVSValue arg = result;
		result = env->Invoke("ConvertToYUY2", AVSValue(&arg, 1)).AsClip();
	}

	return result;
}


PClip FilteredResize::CreateResizeV(PClip clip, double subrange_top, double subrange_height, int target_height,
	ResamplingFunction* func, IScriptEnvironment* env)
{
	const VideoInfo& vi = clip->GetVideoInfo();
	if (subrange_top == 0 && subrange_height == target_height && subrange_height == vi.height) {
		return clip;
	}

	if (subrange_top == int(subrange_top) && subrange_height == target_height
		&& subrange_top >= 0 && subrange_top + subrange_height <= vi.height) {
		const int mask = ((vi.IsYUV() || vi.IsYUVA()) && !vi.IsY()) ? (1 << vi.GetPlaneHeightSubsampling(PLANAR_U)) - 1 : 0;

		if (((int(subrange_top) | int(subrange_height)) & mask) == 0) {
			//return new Crop(0, int(subrange_top), vi.width, int(subrange_height), 0, clip, env);
			AVSValue args[] = { clip, 0, int(subrange_top), vi.width, int(subrange_height), false };
			return env->Invoke("Crop", AVSValue(args, 6)).AsClip();
		}
	}
	return new FilteredResizeV(clip, subrange_top, subrange_height, target_height, func, env);
}


PClip FilteredResize::CreateResize(PClip clip, int target_width, int target_height, const AVSValue* args,
	ResamplingFunction* f, IScriptEnvironment* env)
{
	const VideoInfo& vi = clip->GetVideoInfo();
	const double subrange_left = args[0].AsFloat(0), subrange_top = args[1].AsFloat(0);

	double subrange_width = args[2].AsDblDef(vi.width), subrange_height = args[3].AsDblDef(vi.height);
	// Crop style syntax
	if (subrange_width <= 0.0) subrange_width = vi.width - subrange_left + subrange_width;
	if (subrange_height <= 0.0) subrange_height = vi.height - subrange_top + subrange_height;

	PClip result;
	// ensure that the intermediate area is maximal

	const double area_FirstH = subrange_height * target_width;
	const double area_FirstV = subrange_width * target_height;

	// "minimal area" logic is not necessarily faster because H and V resizers are not the same speed.
	// so we keep the traditional max area logic.
	if (area_FirstH < area_FirstV)
	{
		result = CreateResizeV(clip, subrange_top, subrange_height, target_height, f, env);
		result = CreateResizeH(result, subrange_left, subrange_width, target_width, f, env);
	}
	else
	{
		result = CreateResizeH(clip, subrange_left, subrange_width, target_width, f, env);
		result = CreateResizeV(result, subrange_top, subrange_height, target_height, f, env);
	}
	return result;
}

AVSValue __cdecl FilteredResize::Create_PointResize(AVSValue args, void*, IScriptEnvironment* env)
{
	auto f = PointFilter();
	return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], &f, env);
}


AVSValue __cdecl FilteredResize::Create_BilinearResize(AVSValue args, void*, IScriptEnvironment* env)
{
	auto f = TriangleFilter();
	return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], &f, env);
}


AVSValue __cdecl FilteredResize::Create_BicubicResize(AVSValue args, void*, IScriptEnvironment* env)
{
	auto f = MitchellNetravaliFilter(args[3].AsDblDef(1. / 3.), args[4].AsDblDef(1. / 3.));
	return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[5], &f, env);
}

AVSValue __cdecl FilteredResize::Create_LanczosResize(AVSValue args, void*, IScriptEnvironment* env)
{
	auto f = LanczosFilter(args[7].AsInt(3));
	return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], &f, env);
}

AVSValue __cdecl FilteredResize::Create_Lanczos4Resize(AVSValue args, void*, IScriptEnvironment* env)
{
	auto f = LanczosFilter(4);
	return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], &f, env);
}

AVSValue __cdecl FilteredResize::Create_BlackmanResize(AVSValue args, void*, IScriptEnvironment* env)
{
	auto f = BlackmanFilter(args[7].AsInt(4));
	return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], &f, env);
}

AVSValue __cdecl FilteredResize::Create_Spline16Resize(AVSValue args, void*, IScriptEnvironment* env)
{
	auto f = Spline16Filter();
	return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], &f, env);
}

AVSValue __cdecl FilteredResize::Create_Spline36Resize(AVSValue args, void*, IScriptEnvironment* env)
{
	auto f = Spline36Filter();
	return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], &f, env);
}

AVSValue __cdecl FilteredResize::Create_Spline64Resize(AVSValue args, void*, IScriptEnvironment* env)
{
	auto f = Spline64Filter();
	return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], &f, env);
}

AVSValue __cdecl FilteredResize::Create_GaussianResize(AVSValue args, void*, IScriptEnvironment* env)
{
	auto f = GaussianFilter(args[7].AsFloat(30.0f));
	return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], &f, env);
}

AVSValue __cdecl FilteredResize::Create_SincResize(AVSValue args, void*, IScriptEnvironment* env)
{
	auto f = SincFilter(args[7].AsInt(4));
	return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], &f, env);
}

