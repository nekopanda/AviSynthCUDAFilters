#pragma once

#include <stdint.h>
#include <immintrin.h>

// _mm256_set_epi16‚Íconstexpr‚É‚È‚ç‚È‚¢‚Ì‚ÅAconstexpr‚É‚È‚éŠÖ”‚ð’è‹`
constexpr __m256i const_mm256_setr_epi16(
	short s0, short s1, short s2, short s3, short s4, short s5, short s6, short s7,
	short s8, short s9, short s10, short s11, short s12, short s13, short s14, short s15) {
	return{
		(int8_t)s0, (int8_t)(s0 >> 8),
		(int8_t)s1, (int8_t)(s1 >> 8),
		(int8_t)s2, (int8_t)(s2 >> 8),
		(int8_t)s3, (int8_t)(s3 >> 8),
		(int8_t)s4, (int8_t)(s4 >> 8),
		(int8_t)s5, (int8_t)(s5 >> 8),
		(int8_t)s6, (int8_t)(s6 >> 8),
		(int8_t)s7, (int8_t)(s7 >> 8),
		(int8_t)s8, (int8_t)(s8 >> 8),
		(int8_t)s9, (int8_t)(s9 >> 8),
		(int8_t)s10, (int8_t)(s10 >> 8),
		(int8_t)s11, (int8_t)(s11 >> 8),
		(int8_t)s12, (int8_t)(s12 >> 8),
		(int8_t)s13, (int8_t)(s13 >> 8),
		(int8_t)s14, (int8_t)(s14 >> 8),
		(int8_t)s15, (int8_t)(s15 >> 8)
	};
}
