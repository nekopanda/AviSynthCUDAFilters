
#include <stdint.h>
#include <avisynth.h>

#include <algorithm>

#include <immintrin.h>

// https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2
inline void transpose8_ps(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7) {
	__m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
	__m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
	__t0 = _mm256_unpacklo_ps(row0, row1);
	__t1 = _mm256_unpackhi_ps(row0, row1);
	__t2 = _mm256_unpacklo_ps(row2, row3);
	__t3 = _mm256_unpackhi_ps(row2, row3);
	__t4 = _mm256_unpacklo_ps(row4, row5);
	__t5 = _mm256_unpackhi_ps(row4, row5);
	__t6 = _mm256_unpacklo_ps(row6, row7);
	__t7 = _mm256_unpackhi_ps(row6, row7);
	__tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0));
	__tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2));
	__tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0));
	__tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2));
	__tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0));
	__tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2));
	__tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0));
	__tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2));
	row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
	row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
	row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
	row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
	row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
	row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
	row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
	row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}

// got from the following command in python. (do "from math import *" before)
#define S1    0.19509032201612825f   // sin(1*pi/(2*8))
#define C1    0.9807852804032304f    // cos(1*pi/(2*8))
#define S3    0.5555702330196022f    // sin(3*pi/(2*8))
#define C3    0.8314696123025452f    // cos(3*pi/(2*8))
#define S2S6  1.3065629648763766f    // sqrt(2)*sin(6*pi/(2*8))
#define S2C6  0.5411961001461971f    // sqrt(2)*cos(6*pi/(2*8))
#define S2    1.4142135623730951f    // sqrt(2)

inline void dct_ps(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7)
{
	// stage 1
	auto a0 = _mm256_add_ps(row7, row0);
	auto a1 = _mm256_add_ps(row6, row1);
	auto a2 = _mm256_add_ps(row5, row2);
	auto a3 = _mm256_add_ps(row4, row3);
	auto a4 = _mm256_sub_ps(row3, row4);
	auto a5 = _mm256_sub_ps(row2, row5);
	auto a6 = _mm256_sub_ps(row1, row6);
	auto a7 = _mm256_sub_ps(row0, row7);

	// stage 2 even
	auto b0 = _mm256_add_ps(a3, a0);
	auto b1 = _mm256_add_ps(a2, a1);
	auto b2 = _mm256_sub_ps(a1, a2);
	auto b3 = _mm256_sub_ps(a0, a3);

	// stage 2 odd
	auto b4 = _mm256_add_ps(
		_mm256_mul_ps(_mm256_set1_ps(S3 - C3), a7),
		_mm256_mul_ps(_mm256_set1_ps(C3), _mm256_add_ps(a4, a7)));
	auto b5 = _mm256_add_ps(
		_mm256_mul_ps(_mm256_set1_ps(S1 - C1), a6),
		_mm256_mul_ps(_mm256_set1_ps(C1), _mm256_add_ps(a5, a6)));
	auto b6 = _mm256_add_ps(
		_mm256_mul_ps(_mm256_set1_ps(-(C1 + S1)), a5),
		_mm256_mul_ps(_mm256_set1_ps(C1), _mm256_add_ps(a5, a6)));
	auto b7 = _mm256_add_ps(
		_mm256_mul_ps(_mm256_set1_ps(-(C3 + S3)), a4),
		_mm256_mul_ps(_mm256_set1_ps(C3), _mm256_add_ps(a4, a7)));

	// stage3 even
	auto c0 = _mm256_add_ps(b1, b0);
	auto c1 = _mm256_sub_ps(b0, b1);
	auto c2 = _mm256_add_ps(
		_mm256_mul_ps(_mm256_set1_ps(S2S6 - S2C6), b3),
		_mm256_mul_ps(_mm256_set1_ps(S2C6), _mm256_add_ps(b2, b3)));
	auto c3 = _mm256_add_ps(
		_mm256_mul_ps(_mm256_set1_ps(-(S2C6 + S2S6)), b2), 
		_mm256_mul_ps(_mm256_set1_ps(S2C6), _mm256_add_ps(b2, b3)));

	// stage3 odd
	auto c4 = _mm256_add_ps(b6, b4);
	auto c5 = _mm256_sub_ps(b7, b5);
	auto c6 = _mm256_sub_ps(b4, b6);
	auto c7 = _mm256_add_ps(b5, b7);

	// stage 4 odd
	auto d4 = _mm256_sub_ps(c7, c4);
	auto d5 = _mm256_mul_ps(c5, _mm256_set1_ps(S2));
	auto d6 = _mm256_mul_ps(c6, _mm256_set1_ps(S2));
	auto d7 = _mm256_add_ps(c4, c7);

	// store
	row0 = c0;
	row4 = c1;
	row2 = c2;
	row6 = c3;
	row7 = d4;
	row3 = d5;
	row5 = d6;
	row1 = d7;
}

inline void idct_ps(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7)
{
	auto c0 = row0;
	auto c1 = row4;
	auto c2 = row2;
	auto c3 = row6;
	auto d4 = row7;
	auto d5 = row3;
	auto d6 = row5;
	auto d7 = row1;

	auto c4 = _mm256_sub_ps(d7, d4);
	auto c5 = _mm256_mul_ps(d5, _mm256_set1_ps(S2));
	auto c6 = _mm256_mul_ps(d6, _mm256_set1_ps(S2));
	auto c7 = _mm256_add_ps(d4, d7);

	auto b0 = _mm256_add_ps(c1, c0);
	auto b1 = _mm256_sub_ps(c0, c1);
	auto b2 = _mm256_add_ps(
		_mm256_mul_ps(_mm256_set1_ps(-(S2C6 + S2S6)), c3), 
		_mm256_mul_ps(_mm256_set1_ps(S2C6), _mm256_add_ps(c2, c3)));
	auto b3 = _mm256_add_ps(
		_mm256_mul_ps(_mm256_set1_ps(S2S6 - S2C6), c2),
		_mm256_mul_ps(_mm256_set1_ps(S2C6), _mm256_add_ps(c2, c3)));

	auto b4 = _mm256_add_ps(c6, c4);
	auto b5 = _mm256_sub_ps(c7, c5);
	auto b6 = _mm256_sub_ps(c4, c6);
	auto b7 = _mm256_add_ps(c5, c7);

	auto a0 = _mm256_add_ps(b3, b0);
	auto a1 = _mm256_add_ps(b2, b1);
	auto a2 = _mm256_sub_ps(b1, b2);
	auto a3 = _mm256_sub_ps(b0, b3);

	auto a4 = _mm256_add_ps(
		_mm256_mul_ps(_mm256_set1_ps(-(C3 + S3)), b7),
		_mm256_mul_ps(_mm256_set1_ps(C3), _mm256_add_ps(b4, b7)));
	auto a5 = _mm256_add_ps(
		_mm256_mul_ps(_mm256_set1_ps(-(C1 + S1)), b6),
		_mm256_mul_ps(_mm256_set1_ps(C1), _mm256_add_ps(b5, b6)));
	auto a6 = _mm256_add_ps(
		_mm256_mul_ps(_mm256_set1_ps(S1 - C1), b5), 
		_mm256_mul_ps(_mm256_set1_ps(C1), _mm256_add_ps(b5, b6)));
	auto a7 = _mm256_add_ps(
		_mm256_mul_ps(_mm256_set1_ps(S3 - C3), b4), 
		_mm256_mul_ps(_mm256_set1_ps(C3), _mm256_add_ps(b4, b7)));

	row0 = _mm256_add_ps(a7, a0);
	row1 = _mm256_add_ps(a6, a1);
	row2 = _mm256_add_ps(a5, a2);
	row3 = _mm256_add_ps(a4, a3);
	row4 = _mm256_sub_ps(a3, a4);
	row5 = _mm256_sub_ps(a2, a5);
	row6 = _mm256_sub_ps(a1 ,a6);
	row7 = _mm256_sub_ps(a0, a7);
}

inline __m256 hardthresh_ps(__m256 row, __m256 threshold)
{
	const __m256 signmask = _mm256_set1_ps(-0.0f); // 0x80000000

	// row &= (abs(row) > threshold)
	auto mask = _mm256_cmp_ps(_mm256_andnot_ps(signmask, row), threshold, _CMP_GT_OS);
	return _mm256_and_ps(row, mask);
}

inline void hardthresh_avx(float threshold_,
	__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7)
{
	auto threshold = _mm256_set1_ps(threshold_);

	__m256 row0orig = row0;
	
	row0 = hardthresh_ps(row0, threshold);
	row1 = hardthresh_ps(row1, threshold);
	row2 = hardthresh_ps(row2, threshold);
	row3 = hardthresh_ps(row3, threshold);
	row4 = hardthresh_ps(row4, threshold);
	row5 = hardthresh_ps(row5, threshold);
	row6 = hardthresh_ps(row6, threshold);
	row7 = hardthresh_ps(row7, threshold);

	// [0]だけもとの値に戻す
	row0 = _mm256_blend_ps(row0, row0orig, 1);
}

inline __m256 softthresh_ps(__m256 row, __m256 threshold)
{
	const __m256 signmask = _mm256_set1_ps(-0.0f); // 0x80000000
	
	// sign(row) | max(0, abs(row) - threshold)
	auto a = _mm256_max_ps(_mm256_setzero_ps(),
		_mm256_sub_ps(_mm256_andnot_ps(signmask, row), threshold));
	return _mm256_or_ps(a, _mm256_and_ps(row, signmask));
}

inline void softthresh_avx(float threshold_,
	__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7)
{
	auto threshold = _mm256_set1_ps(threshold_);

	__m256 row0orig = row0;

	row0 = softthresh_ps(row0, threshold);
	row1 = softthresh_ps(row1, threshold);
	row2 = softthresh_ps(row2, threshold);
	row3 = softthresh_ps(row3, threshold);
	row4 = softthresh_ps(row4, threshold);
	row5 = softthresh_ps(row5, threshold);
	row6 = softthresh_ps(row6, threshold);
	row7 = softthresh_ps(row7, threshold);

	// [0]だけもとの値に戻す
	row0 = _mm256_blend_ps(row0, row0orig, 1);
}

inline void add_to_block_avx(uint16_t* dst, __m256 row, __m256 half, int shift, __m256i maxv)
{
	// a = (int)(row + half) >> shift
	auto a = _mm256_sra_epi32(
		_mm256_cvttps_epi32(_mm256_add_ps(row, half)),
		_mm_cvtsi32_si128(shift));
	// b = clamp(a, 0, maxv)
	auto b = _mm256_min_epi32(_mm256_max_epi32(a, _mm256_setzero_si256()), maxv);
	// c = (short)b
	auto c = _mm_packus_epi32(_mm256_extractf128_si256(b, 0), _mm256_extractf128_si256(b, 1));
	// *dst += c
	_mm_storeu_si128((__m128i*)dst, _mm_add_epi32(_mm_loadu_si128((__m128i*)dst), c));
}

inline void add_block_avx(uint16_t* dst, int dst_pitch, float half_, int shift, int maxv_,
	__m256 row0, __m256 row1, __m256 row2, __m256 row3, __m256 row4, __m256 row5, __m256 row6, __m256 row7)
{
	auto half = _mm256_set1_ps(half_);
	auto maxv = _mm256_set1_epi32(maxv_);
	add_to_block_avx(&dst[dst_pitch * 0], row0, half, shift, maxv);
	add_to_block_avx(&dst[dst_pitch * 1], row1, half, shift, maxv);
	add_to_block_avx(&dst[dst_pitch * 2], row2, half, shift, maxv);
	add_to_block_avx(&dst[dst_pitch * 3], row3, half, shift, maxv);
	add_to_block_avx(&dst[dst_pitch * 4], row4, half, shift, maxv);
	add_to_block_avx(&dst[dst_pitch * 5], row5, half, shift, maxv);
	add_to_block_avx(&dst[dst_pitch * 6], row6, half, shift, maxv);
	add_to_block_avx(&dst[dst_pitch * 7], row7, half, shift, maxv);
}

__forceinline void cpu_deblock_kernel_avx(uint16_t* dst, int dst_pitch, float thresh, bool is_soft, float half, int shift, int maxv,
	__m256 row0, __m256 row1, __m256 row2, __m256 row3, __m256 row4, __m256 row5, __m256 row6, __m256 row7)
{
	// dct
	dct_ps(row0, row1, row2, row3, row4, row5, row6, row7);
	transpose8_ps(row0, row1, row2, row3, row4, row5, row6, row7);
	dct_ps(row0, row1, row2, row3, row4, row5, row6, row7);

	// 転置された状態だけど[0]の位置は同じなので問題ない

	// requantize
	if (is_soft) {
		softthresh_avx(thresh, row0, row1, row2, row3, row4, row5, row6, row7);
	}
	else {
		hardthresh_avx(thresh, row0, row1, row2, row3, row4, row5, row6, row7);
	}

	// idct
	idct_ps(row0, row1, row2, row3, row4, row5, row6, row7);
	transpose8_ps(row0, row1, row2, row3, row4, row5, row6, row7);
	idct_ps(row0, row1, row2, row3, row4, row5, row6, row7);

	add_block_avx(dst, dst_pitch, half, shift, maxv, row0, row1, row2, row3, row4, row5, row6, row7);
}

inline __m256 load_to_float_avx(const uint8_t* src) {
	return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(*(const int64_t*)src)));
}

inline __m256 load_to_float_avx(const uint16_t* src) {
	return _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)src)));
}

template <typename pixel_t>
void cpu_deblock_kernel_avx(const pixel_t* src, int src_pitch,
	uint16_t* dst, int dst_pitch, float thresh, bool is_soft, float half, int shift, int maxv)
{
	__m256 row0 = load_to_float_avx(&src[src_pitch * 0]);
	__m256 row1 = load_to_float_avx(&src[src_pitch * 1]);
	__m256 row2 = load_to_float_avx(&src[src_pitch * 2]);
	__m256 row3 = load_to_float_avx(&src[src_pitch * 3]);
	__m256 row4 = load_to_float_avx(&src[src_pitch * 4]);
	__m256 row5 = load_to_float_avx(&src[src_pitch * 5]);
	__m256 row6 = load_to_float_avx(&src[src_pitch * 6]);
	__m256 row7 = load_to_float_avx(&src[src_pitch * 7]);

	cpu_deblock_kernel_avx(
		dst, dst_pitch, thresh, is_soft, half, shift, maxv,
		row0, row1, row2, row3, row4, row5, row6, row7);
}

template void cpu_deblock_kernel_avx<uint8_t>(const uint8_t* src, int src_pitch,
	uint16_t* dst, int dst_pitch, float thresh, bool is_soft, float half, int shift, int maxv);
template void cpu_deblock_kernel_avx<uint16_t>(const uint16_t* src, int src_pitch,
	uint16_t* dst, int dst_pitch, float thresh, bool is_soft, float half, int shift, int maxv);


const __m256i g_lditherv[8] = {
	_mm256_set_epi16( 0,  48,  12,  60,   3,  51,  15,  63,  0,  48,  12,  60,   3,  51,  15,  63),
	_mm256_set_epi16(32,  16,  44,  28,  35,  19,  47,  31, 32,  16,  44,  28,  35,  19,  47,  31),
	_mm256_set_epi16( 8,  56,   4,  52,  11,  59,   7,  55,  8,  56,   4,  52,  11,  59,   7,  55),
	_mm256_set_epi16(40,  24,  36,  20,  43,  27,  39,  23, 40,  24,  36,  20,  43,  27,  39,  23),
	_mm256_set_epi16( 2,  50,  14,  62,   1,  49,  13,  61,  2,  50,  14,  62,   1,  49,  13,  61),
	_mm256_set_epi16(34,  18,  46,  30,  33,  17,  45,  29, 34,  18,  46,  30,  33,  17,  45,  29),
	_mm256_set_epi16(10,  58,   6,  54,   9,  57,   5,  53, 10,  58,   6,  54,   9,  57,   5,  53),
	_mm256_set_epi16(42,  26,  38,  22,  41,  25,  37,  21, 42,  26,  38,  22,  41,  25,  37,  21)
};

template <int shift>
inline __m256i make_store_value_avx(const uint16_t* tmp, int y, int maxv) {
	enum { SHIFT = shift - 6 };
	auto t = _mm256_loadu_si256((const __m256i*)tmp);
#pragma warning(push)
#pragma warning(disable:4556)
	// (tmp >> SHIFT)
	if (SHIFT > 0) {
		t = _mm256_srli_epi16(t, SHIFT);
	}
	else if(SHIFT < 0) {
		t = _mm256_slli_epi16(t, -SHIFT);
	}
#pragma warning(pop)
	// min((t + dither) >> 6, maxv)
	return _mm256_min_epu16(
		_mm256_srli_epi16(_mm256_adds_epu16(t, g_lditherv[y]), 6),
		_mm256_set1_epi16(maxv));
}

void store_u16_to(uint8_t* dst, __m256i u16v) {
	_mm_storeu_si128((__m128i*)dst,
		_mm_packus_epi16(_mm256_extractf128_si256(u16v, 0), _mm256_extractf128_si256(u16v, 1)));
}

void store_u16_to(uint16_t* dst, __m256i u16v) {
	_mm256_storeu_si256((__m256i*)dst, u16v);
}

// height <= 8
template <typename pixel_t, int shift>
void cpu_store_slice_avx_tmpl(
	int width, int height, pixel_t* dst, int dst_pitch,
	const uint16_t* tmp, int tmp_pitch, int maxv)
{
	for (int y = 0; y < height; ++y) {
		int x = 0;
		for (; x <= (width - 16); x += 16) {
			store_u16_to(&dst[x + y * dst_pitch],
				make_store_value_avx<shift>(&tmp[x + y * tmp_pitch], y, maxv));
		}
		if (x < width) {
			auto t = make_store_value_avx<shift>(&tmp[x + y * tmp_pitch], y, maxv);
			for (int i = 0; x < width; ++x, ++i) {
				dst[x + y * dst_pitch] = (pixel_t)t.m256i_u16[i];
			}
		}
	}
}

template <typename pixel_t>
void cpu_store_slice_avx(
	int width, int height, pixel_t* dst, int dst_pitch,
	const uint16_t* tmp, int tmp_pitch, int shift, int maxv)
{
	static void(*table[])(
		int width, int height, pixel_t* dst, int dst_pitch,
		const uint16_t* tmp, int tmp_pitch, int maxv) = {
		cpu_store_slice_avx_tmpl<pixel_t, 0>,
		cpu_store_slice_avx_tmpl<pixel_t, 1>,
		cpu_store_slice_avx_tmpl<pixel_t, 2>,
		cpu_store_slice_avx_tmpl<pixel_t, 3>,
		cpu_store_slice_avx_tmpl<pixel_t, 4>,
		cpu_store_slice_avx_tmpl<pixel_t, 5>,
		cpu_store_slice_avx_tmpl<pixel_t, 6>,
		cpu_store_slice_avx_tmpl<pixel_t, 7>,
		cpu_store_slice_avx_tmpl<pixel_t, 8>
	};
	table[shift](width, height, dst, dst_pitch, tmp, tmp_pitch, maxv);
}

template void cpu_store_slice_avx<uint8_t>(
	int width, int height, uint8_t* dst, int dst_pitch,
	const uint16_t* tmp, int tmp_pitch, int shift, int maxv);
template void cpu_store_slice_avx<uint16_t>(
	int width, int height, uint16_t* dst, int dst_pitch,
	const uint16_t* tmp, int tmp_pitch, int shift, int maxv);
