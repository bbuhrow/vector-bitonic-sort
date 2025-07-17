// MIT License
// 
// Copyright (c) 2025 Ben Buhrow
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


// fast bitonic sort using AVX-512 and test code to find
// collisions of N-bit keys in many contiguous medium-
// sized lists of random keys.
// Written by Ben Buhrow, July 2025
// example compile line:
// clang -O2 -g -march=icelake-client vec_bitonic_sort.c -o vecsort
// REQUIRES AVX-512 or will fail with illegal instruction 


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <immintrin.h>
#include <omp.h>



#if defined(WIN32) || defined(_WIN64)
	#define WIN32_LEAN_AND_MEAN

	#include <windows.h>
#else
	#include <fcntl.h>
	#include <unistd.h>
	#include <sys/resource.h>
#endif

typedef uint8_t uint8;
typedef uint32_t uint32;
typedef uint64_t uint64;

/* #define HAVE_PROF */
#ifdef HAVE_PROF
#define SHOW_PROF __attribute__((noinline))
#else
#define SHOW_PROF /* nothing */
#endif


int qcomp_uint64(const void *x, const void *y)
{
	uint64_t *xx = (uint64_t *)x;
	uint64_t *yy = (uint64_t *)y;
	
	return (*xx > *yy) - (*xx < *yy);
	
	
	if (*xx > *yy)
		return 1;
	else if (*xx == *yy)
		return 0;
	else
		return -1;
}

int qcomp_uint32(const void *x, const void *y)
{
	uint32_t *xx = (uint32_t *)x;
	uint32_t *yy = (uint32_t *)y;
	
	return (*xx > *yy) - (*xx < *yy);
	
	
	if (*xx > *yy)
		return 1;
	else if (*xx == *yy)
		return 0;
	else
		return -1;
}

static void *
aligned_malloc(size_t len, uint32 align) {

	void *ptr, *aligned_ptr;
	unsigned long addr;

	ptr = malloc(len+align);

	 /* offset to next ALIGN-byte boundary */

	addr = (unsigned long)ptr;				
	addr = align - (addr & (align-1));
	aligned_ptr = (void *)((uint8 *)ptr + addr);

	*( (void **)aligned_ptr - 1 ) = ptr;
	return aligned_ptr;
}


static void
aligned_free(void *newptr) {

	void *ptr;

	if (newptr == NULL) 
		return;
	ptr = *( (void **)newptr - 1 );
	free(ptr);
}


double
get_cpu_time(void) {

#if defined(WIN32) || defined(_WIN64)
	FILETIME create_time = {0, 0};
	FILETIME exit_time = {0, 0};
	FILETIME kernel_time = {0, 0};
	FILETIME user_time = {0, 0};

	GetThreadTimes(GetCurrentThread(),
			&create_time,
			&exit_time,
			&kernel_time,
			&user_time);

	return ((uint64)user_time.dwHighDateTime << 32 | 
	               user_time.dwLowDateTime) / 10000000.0;
#else
	struct rusage r_usage;

	#if 0 /* use for linux 2.6.26+ */
	getrusage(RUSAGE_THREAD, &r_usage);
	#else
	getrusage(RUSAGE_SELF, &r_usage);
	#endif

	return ((uint64)r_usage.ru_utime.tv_sec * 1000000 +
	               r_usage.ru_utime.tv_usec) / 1000000.0;
#endif
}


static uint32 
get_rand(uint32 *rand_seed, uint32 *rand_carry) {
   
	/* A multiply-with-carry generator by George Marsaglia.
	   The period is about 2^63. */

	#define RAND_MULT 2131995753

	uint64 temp;

	temp = (uint64)(*rand_seed) * 
		       (uint64)RAND_MULT + 
		       (uint64)(*rand_carry);
	*rand_seed = (uint32)temp;
	*rand_carry = (uint32)(temp >> 32);
	return (uint32)temp;
}

uint32_t my_clz32(uint64_t n)
{
#if (INLINE_ASM && defined(__x86_64__))
#ifdef __BMI1__
	uint32_t t;
	asm(" lzcnt %1, %0\n": "=r"(t) : "r"(n) : "flags");
	return t;
#else
	if (n)
		return __builtin_clz(n);
	return 32;
#endif
#else
#if defined(__GNUC__)
	if (n)
		return __builtin_clz(n);
	return 32;
#else
	if (n == 0)
		return 32;
	uint32_t r = 0;
	if ((n & (0xFFFFull << 16)) == 0)
		r += 16, n <<= 16;
	if ((n & (0xFFull << 24)) == 0)
		r += 8, n <<= 8;
	if ((n & (0xFull << 28)) == 0)
		r += 4, n <<= 4;
	if ((n & (0x3ull << 30)) == 0)
		r += 2, n <<= 2;
	if ((n & (0x1ull << 31)) == 0)
		r += 1;
	return r;
#endif
#endif
}

// intrinsics for swapping N-bit chunks of data within a 512-bit vector
// that use immediates (faster and fewer registers than needing to load index vectors)
#define SWAP32(x) _mm512_shuffle_epi32((x), 0xB1)
#define SWAP64(x) _mm512_shuffle_epi32((x), 0x4E)
#define SWAP128(x) _mm512_permutex_epi64((x), 0x4E)
#define SWAP256(x) _mm512_shuffle_i64x2((x), (x), 0x4E)

// intrinsics for swapping N-bit chunks of data within a 512-bit vector, with N < 32.
// these require index vectors and the additional extension AVX-512BW
static __m512i swap8bit_idx;
static __m512i swap16bit_idx;

#define SWAP8(x) _mm512_shuffle_epi8((x), swap8bit_idx)
#define SWAP16(x) _mm512_shuffle_epi8((x), swap16bit_idx)

#define CMPGT(v1, v2, v3, v4, v5, v6, v7, v8) \
	m1 = _mm512_cmpgt_epu32_mask(v1, dv1_swap);  \
	m2 = _mm512_cmpgt_epu32_mask(v2, dv2_swap);  \
	m3 = _mm512_cmpgt_epu32_mask(v3, dv3_swap);  \
	m4 = _mm512_cmpgt_epu32_mask(v4, dv4_swap);  \
	m5 = _mm512_cmpgt_epu32_mask(v5, dv5_swap);  \
	m6 = _mm512_cmpgt_epu32_mask(v6, dv6_swap);  \
	m7 = _mm512_cmpgt_epu32_mask(v7, dv7_swap);  \
	m8 = _mm512_cmpgt_epu32_mask(v8, dv8_swap);
	
#define CMPLT(v1, v2, v3, v4, v5, v6, v7, v8) \
	m1 = _mm512_cmplt_epu32_mask(v1, dv1_swap);  \
	m2 = _mm512_cmplt_epu32_mask(v2, dv2_swap);  \
	m3 = _mm512_cmplt_epu32_mask(v3, dv3_swap);  \
	m4 = _mm512_cmplt_epu32_mask(v4, dv4_swap);  \
	m5 = _mm512_cmplt_epu32_mask(v5, dv5_swap);  \
	m6 = _mm512_cmplt_epu32_mask(v6, dv6_swap);  \
	m7 = _mm512_cmplt_epu32_mask(v7, dv7_swap);  \
	m8 = _mm512_cmplt_epu32_mask(v8, dv8_swap);
		
#define CMP(v1, v2, v3, v4, v5, v6, v7, v8, c1, c2, c3, c4, c5, c6, c7, c8) \
	m1 = _mm512_cmp_epu32_mask(v1, dv1_swap, c1);  \
	m2 = _mm512_cmp_epu32_mask(v2, dv2_swap, c2);  \
	m3 = _mm512_cmp_epu32_mask(v3, dv3_swap, c3);  \
	m4 = _mm512_cmp_epu32_mask(v4, dv4_swap, c4);  \
	m5 = _mm512_cmp_epu32_mask(v5, dv5_swap, c5);  \
	m6 = _mm512_cmp_epu32_mask(v6, dv6_swap, c6);  \
	m7 = _mm512_cmp_epu32_mask(v7, dv7_swap, c7);  \
	m8 = _mm512_cmp_epu32_mask(v8, dv8_swap, c8);
	
#define BLENDMASK(M, v1, v2, v3, v4, v5, v6, v7, v8) \
	v1 = _mm512_mask_blend_epi32(m1 ^ (M), v1, dv1_swap);  \
	v2 = _mm512_mask_blend_epi32(m2 ^ (M), v2, dv2_swap);  \
	v3 = _mm512_mask_blend_epi32(m3 ^ (M), v3, dv3_swap);  \
	v4 = _mm512_mask_blend_epi32(m4 ^ (M), v4, dv4_swap);  \
	v5 = _mm512_mask_blend_epi32(m5 ^ (M), v5, dv5_swap);  \
	v6 = _mm512_mask_blend_epi32(m6 ^ (M), v6, dv6_swap);  \
	v7 = _mm512_mask_blend_epi32(m7 ^ (M), v7, dv7_swap);  \
	v8 = _mm512_mask_blend_epi32(m8 ^ (M), v8, dv8_swap);

#define MINMAX(m1, m2, v1, v2, v3, v4, v5, v6, v7, v8) \
	t1 = _mm512_mask_max_epu32(v1, m1, v1, dv1_swap);		\
	t2 = _mm512_mask_max_epu32(v2, m1, v2, dv2_swap);     \
	t3 = _mm512_mask_max_epu32(v3, m1, v3, dv3_swap);     \
	t4 = _mm512_mask_max_epu32(v4, m1, v4, dv4_swap);     \
	t5 = _mm512_mask_max_epu32(v5, m1, v5, dv5_swap);     \
	t6 = _mm512_mask_max_epu32(v6, m1, v6, dv6_swap);     \
	t7 = _mm512_mask_max_epu32(v7, m1, v7, dv7_swap);     \
	t8 = _mm512_mask_max_epu32(v8, m1, v8, dv8_swap);     \
	v1 = _mm512_mask_min_epu32(t1, m2, v1, dv1_swap);     \
	v2 = _mm512_mask_min_epu32(t2, m2, v2, dv2_swap);     \
	v3 = _mm512_mask_min_epu32(t3, m2, v3, dv3_swap);     \
	v4 = _mm512_mask_min_epu32(t4, m2, v4, dv4_swap);     \
	v5 = _mm512_mask_min_epu32(t5, m2, v5, dv5_swap);     \
	v6 = _mm512_mask_min_epu32(t6, m2, v6, dv6_swap);     \
	v7 = _mm512_mask_min_epu32(t7, m2, v7, dv7_swap);     \
	v8 = _mm512_mask_min_epu32(t8, m2, v8, dv8_swap); 
	
#define MINMAX_alt1(m1, m2, v1, v2, v3, v4, v5, v6, v7, v8) \
	t1 = _mm512_mask_max_epu32(v1, m1, v1, dv1_swap);		\
	t2 = _mm512_mask_min_epu32(v2, m1, v2, dv2_swap);     \
	t3 = _mm512_mask_max_epu32(v3, m1, v3, dv3_swap);     \
	t4 = _mm512_mask_min_epu32(v4, m1, v4, dv4_swap);     \
	t5 = _mm512_mask_max_epu32(v5, m1, v5, dv5_swap);     \
	t6 = _mm512_mask_min_epu32(v6, m1, v6, dv6_swap);     \
	t7 = _mm512_mask_max_epu32(v7, m1, v7, dv7_swap);     \
	t8 = _mm512_mask_min_epu32(v8, m1, v8, dv8_swap);     \
	v1 = _mm512_mask_min_epu32(t1, m2, v1, dv1_swap);     \
	v2 = _mm512_mask_max_epu32(t2, m2, v2, dv2_swap);     \
	v3 = _mm512_mask_min_epu32(t3, m2, v3, dv3_swap);     \
	v4 = _mm512_mask_max_epu32(t4, m2, v4, dv4_swap);     \
	v5 = _mm512_mask_min_epu32(t5, m2, v5, dv5_swap);     \
	v6 = _mm512_mask_max_epu32(t6, m2, v6, dv6_swap);     \
	v7 = _mm512_mask_min_epu32(t7, m2, v7, dv7_swap);     \
	v8 = _mm512_mask_max_epu32(t8, m2, v8, dv8_swap); 
	
#define MINMAX_alt2(m1, m2, v1, v2, v3, v4, v5, v6, v7, v8) \
	t1 = _mm512_mask_max_epu32(v1, m1, v1, dv1_swap);		\
	t2 = _mm512_mask_max_epu32(v2, m1, v2, dv2_swap);     \
	t3 = _mm512_mask_min_epu32(v3, m1, v3, dv3_swap);     \
	t4 = _mm512_mask_min_epu32(v4, m1, v4, dv4_swap);     \
	t5 = _mm512_mask_max_epu32(v5, m1, v5, dv5_swap);     \
	t6 = _mm512_mask_max_epu32(v6, m1, v6, dv6_swap);     \
	t7 = _mm512_mask_min_epu32(v7, m1, v7, dv7_swap);     \
	t8 = _mm512_mask_min_epu32(v8, m1, v8, dv8_swap);     \
	v1 = _mm512_mask_min_epu32(t1, m2, v1, dv1_swap);     \
	v2 = _mm512_mask_min_epu32(t2, m2, v2, dv2_swap);     \
	v3 = _mm512_mask_max_epu32(t3, m2, v3, dv3_swap);     \
	v4 = _mm512_mask_max_epu32(t4, m2, v4, dv4_swap);     \
	v5 = _mm512_mask_min_epu32(t5, m2, v5, dv5_swap);     \
	v6 = _mm512_mask_min_epu32(t6, m2, v6, dv6_swap);     \
	v7 = _mm512_mask_max_epu32(t7, m2, v7, dv7_swap);     \
	v8 = _mm512_mask_max_epu32(t8, m2, v8, dv8_swap); 
	
#define MINMAX_alt4(m1, m2, v1, v2, v3, v4, v5, v6, v7, v8) \
	t1 = _mm512_mask_max_epu32(v1, m1, v1, dv1_swap);		\
	t2 = _mm512_mask_max_epu32(v2, m1, v2, dv2_swap);     \
	t3 = _mm512_mask_max_epu32(v3, m1, v3, dv3_swap);     \
	t4 = _mm512_mask_max_epu32(v4, m1, v4, dv4_swap);     \
	t5 = _mm512_mask_min_epu32(v5, m1, v5, dv5_swap);     \
	t6 = _mm512_mask_min_epu32(v6, m1, v6, dv6_swap);     \
	t7 = _mm512_mask_min_epu32(v7, m1, v7, dv7_swap);     \
	t8 = _mm512_mask_min_epu32(v8, m1, v8, dv8_swap);     \
	v1 = _mm512_mask_min_epu32(t1, m2, v1, dv1_swap);     \
	v2 = _mm512_mask_min_epu32(t2, m2, v2, dv2_swap);     \
	v3 = _mm512_mask_min_epu32(t3, m2, v3, dv3_swap);     \
	v4 = _mm512_mask_min_epu32(t4, m2, v4, dv4_swap);     \
	v5 = _mm512_mask_max_epu32(t5, m2, v5, dv5_swap);     \
	v6 = _mm512_mask_max_epu32(t6, m2, v6, dv6_swap);     \
	v7 = _mm512_mask_max_epu32(t7, m2, v7, dv7_swap);     \
	v8 = _mm512_mask_max_epu32(t8, m2, v8, dv8_swap); 

#define CMPGT64(v1, v2, v3, v4, v5, v6, v7, v8) \
	m1 = _mm512_cmpgt_epu64_mask(v1, dv1_swap);  \
	m2 = _mm512_cmpgt_epu64_mask(v2, dv2_swap);  \
	m3 = _mm512_cmpgt_epu64_mask(v3, dv3_swap);  \
	m4 = _mm512_cmpgt_epu64_mask(v4, dv4_swap);  \
	m5 = _mm512_cmpgt_epu64_mask(v5, dv5_swap);  \
	m6 = _mm512_cmpgt_epu64_mask(v6, dv6_swap);  \
	m7 = _mm512_cmpgt_epu64_mask(v7, dv7_swap);  \
	m8 = _mm512_cmpgt_epu64_mask(v8, dv8_swap);
	
#define CMPLT64(v1, v2, v3, v4, v5, v6, v7, v8) \
	m1 = _mm512_cmplt_epu64_mask(v1, dv1_swap);  \
	m2 = _mm512_cmplt_epu64_mask(v2, dv2_swap);  \
	m3 = _mm512_cmplt_epu64_mask(v3, dv3_swap);  \
	m4 = _mm512_cmplt_epu64_mask(v4, dv4_swap);  \
	m5 = _mm512_cmplt_epu64_mask(v5, dv5_swap);  \
	m6 = _mm512_cmplt_epu64_mask(v6, dv6_swap);  \
	m7 = _mm512_cmplt_epu64_mask(v7, dv7_swap);  \
	m8 = _mm512_cmplt_epu64_mask(v8, dv8_swap);
		
#define CMP64(v1, v2, v3, v4, v5, v6, v7, v8, c1, c2, c3, c4, c5, c6, c7, c8) \
	m1 = _mm512_cmp_epu64_mask(v1, dv1_swap, c1);  \
	m2 = _mm512_cmp_epu64_mask(v2, dv2_swap, c2);  \
	m3 = _mm512_cmp_epu64_mask(v3, dv3_swap, c3);  \
	m4 = _mm512_cmp_epu64_mask(v4, dv4_swap, c4);  \
	m5 = _mm512_cmp_epu64_mask(v5, dv5_swap, c5);  \
	m6 = _mm512_cmp_epu64_mask(v6, dv6_swap, c6);  \
	m7 = _mm512_cmp_epu64_mask(v7, dv7_swap, c7);  \
	m8 = _mm512_cmp_epu64_mask(v8, dv8_swap, c8);
	
#define BLENDMASK64(M, v1, v2, v3, v4, v5, v6, v7, v8) \
	v1 = _mm512_mask_blend_epi64(m1 ^ (M), v1, dv1_swap);  \
	v2 = _mm512_mask_blend_epi64(m2 ^ (M), v2, dv2_swap);  \
	v3 = _mm512_mask_blend_epi64(m3 ^ (M), v3, dv3_swap);  \
	v4 = _mm512_mask_blend_epi64(m4 ^ (M), v4, dv4_swap);  \
	v5 = _mm512_mask_blend_epi64(m5 ^ (M), v5, dv5_swap);  \
	v6 = _mm512_mask_blend_epi64(m6 ^ (M), v6, dv6_swap);  \
	v7 = _mm512_mask_blend_epi64(m7 ^ (M), v7, dv7_swap);  \
	v8 = _mm512_mask_blend_epi64(m8 ^ (M), v8, dv8_swap);

#define MINMAX64(m1, m2, v1, v2, v3, v4, v5, v6, v7, v8) \
	t1 = _mm512_mask_max_epu64(v1, m1, v1, dv1_swap);		\
	t2 = _mm512_mask_max_epu64(v2, m1, v2, dv2_swap);     \
	t3 = _mm512_mask_max_epu64(v3, m1, v3, dv3_swap);     \
	t4 = _mm512_mask_max_epu64(v4, m1, v4, dv4_swap);     \
	t5 = _mm512_mask_max_epu64(v5, m1, v5, dv5_swap);     \
	t6 = _mm512_mask_max_epu64(v6, m1, v6, dv6_swap);     \
	t7 = _mm512_mask_max_epu64(v7, m1, v7, dv7_swap);     \
	t8 = _mm512_mask_max_epu64(v8, m1, v8, dv8_swap);     \
	v1 = _mm512_mask_min_epu64(t1, m2, v1, dv1_swap);     \
	v2 = _mm512_mask_min_epu64(t2, m2, v2, dv2_swap);     \
	v3 = _mm512_mask_min_epu64(t3, m2, v3, dv3_swap);     \
	v4 = _mm512_mask_min_epu64(t4, m2, v4, dv4_swap);     \
	v5 = _mm512_mask_min_epu64(t5, m2, v5, dv5_swap);     \
	v6 = _mm512_mask_min_epu64(t6, m2, v6, dv6_swap);     \
	v7 = _mm512_mask_min_epu64(t7, m2, v7, dv7_swap);     \
	v8 = _mm512_mask_min_epu64(t8, m2, v8, dv8_swap); 
	
#define MINMAX64_alt1(m1, m2, v1, v2, v3, v4, v5, v6, v7, v8) \
	t1 = _mm512_mask_max_epu64(v1, m1, v1, dv1_swap);		\
	t2 = _mm512_mask_min_epu64(v2, m1, v2, dv2_swap);     \
	t3 = _mm512_mask_max_epu64(v3, m1, v3, dv3_swap);     \
	t4 = _mm512_mask_min_epu64(v4, m1, v4, dv4_swap);     \
	t5 = _mm512_mask_max_epu64(v5, m1, v5, dv5_swap);     \
	t6 = _mm512_mask_min_epu64(v6, m1, v6, dv6_swap);     \
	t7 = _mm512_mask_max_epu64(v7, m1, v7, dv7_swap);     \
	t8 = _mm512_mask_min_epu64(v8, m1, v8, dv8_swap);     \
	v1 = _mm512_mask_min_epu64(t1, m2, v1, dv1_swap);     \
	v2 = _mm512_mask_max_epu64(t2, m2, v2, dv2_swap);     \
	v3 = _mm512_mask_min_epu64(t3, m2, v3, dv3_swap);     \
	v4 = _mm512_mask_max_epu64(t4, m2, v4, dv4_swap);     \
	v5 = _mm512_mask_min_epu64(t5, m2, v5, dv5_swap);     \
	v6 = _mm512_mask_max_epu64(t6, m2, v6, dv6_swap);     \
	v7 = _mm512_mask_min_epu64(t7, m2, v7, dv7_swap);     \
	v8 = _mm512_mask_max_epu64(t8, m2, v8, dv8_swap); 
	
#define MINMAX64_alt2(m1, m2, v1, v2, v3, v4, v5, v6, v7, v8) \
	t1 = _mm512_mask_max_epu64(v1, m1, v1, dv1_swap);		\
	t2 = _mm512_mask_max_epu64(v2, m1, v2, dv2_swap);     \
	t3 = _mm512_mask_min_epu64(v3, m1, v3, dv3_swap);     \
	t4 = _mm512_mask_min_epu64(v4, m1, v4, dv4_swap);     \
	t5 = _mm512_mask_max_epu64(v5, m1, v5, dv5_swap);     \
	t6 = _mm512_mask_max_epu64(v6, m1, v6, dv6_swap);     \
	t7 = _mm512_mask_min_epu64(v7, m1, v7, dv7_swap);     \
	t8 = _mm512_mask_min_epu64(v8, m1, v8, dv8_swap);     \
	v1 = _mm512_mask_min_epu64(t1, m2, v1, dv1_swap);     \
	v2 = _mm512_mask_min_epu64(t2, m2, v2, dv2_swap);     \
	v3 = _mm512_mask_max_epu64(t3, m2, v3, dv3_swap);     \
	v4 = _mm512_mask_max_epu64(t4, m2, v4, dv4_swap);     \
	v5 = _mm512_mask_min_epu64(t5, m2, v5, dv5_swap);     \
	v6 = _mm512_mask_min_epu64(t6, m2, v6, dv6_swap);     \
	v7 = _mm512_mask_max_epu64(t7, m2, v7, dv7_swap);     \
	v8 = _mm512_mask_max_epu64(t8, m2, v8, dv8_swap); 
	
#define MINMAX64_alt4(m1, m2, v1, v2, v3, v4, v5, v6, v7, v8) \
	t1 = _mm512_mask_max_epu64(v1, m1, v1, dv1_swap);		\
	t2 = _mm512_mask_max_epu64(v2, m1, v2, dv2_swap);     \
	t3 = _mm512_mask_max_epu64(v3, m1, v3, dv3_swap);     \
	t4 = _mm512_mask_max_epu64(v4, m1, v4, dv4_swap);     \
	t5 = _mm512_mask_min_epu64(v5, m1, v5, dv5_swap);     \
	t6 = _mm512_mask_min_epu64(v6, m1, v6, dv6_swap);     \
	t7 = _mm512_mask_min_epu64(v7, m1, v7, dv7_swap);     \
	t8 = _mm512_mask_min_epu64(v8, m1, v8, dv8_swap);     \
	v1 = _mm512_mask_min_epu64(t1, m2, v1, dv1_swap);     \
	v2 = _mm512_mask_min_epu64(t2, m2, v2, dv2_swap);     \
	v3 = _mm512_mask_min_epu64(t3, m2, v3, dv3_swap);     \
	v4 = _mm512_mask_min_epu64(t4, m2, v4, dv4_swap);     \
	v5 = _mm512_mask_max_epu64(t5, m2, v5, dv5_swap);     \
	v6 = _mm512_mask_max_epu64(t6, m2, v6, dv6_swap);     \
	v7 = _mm512_mask_max_epu64(t7, m2, v7, dv7_swap);     \
	v8 = _mm512_mask_max_epu64(t8, m2, v8, dv8_swap); 


void print128(__m512i i1, __m512i i2, __m512i i3, __m512i i4,
	__m512i i5, __m512i i6, __m512i i7, __m512i i8)
{
	uint32_t t[16];
	int i;
	
	_mm512_storeu_epi32(t, i1);
	for (i = 0; i < 16; i++)
	{
		printf("%08u ", t[i]);
	}
	printf("\n");
	
	_mm512_storeu_epi32(t, i2);
	for (i = 0; i < 16; i++)
	{
		printf("%08u ", t[i]);
	}
	printf("\n");
	
	_mm512_storeu_epi32(t, i3);
	for (i = 0; i < 16; i++)
	{
		printf("%08u ", t[i]);
	}
	printf("\n");
	
	_mm512_storeu_epi32(t, i4);
	for (i = 0; i < 16; i++)
	{
		printf("%08u ", t[i]);
	}
	printf("\n");
	
	_mm512_storeu_epi32(t, i5);
	for (i = 0; i < 16; i++)
	{
		printf("%08u ", t[i]);
	}
	printf("\n");
	
	_mm512_storeu_epi32(t, i6);
	for (i = 0; i < 16; i++)
	{
		printf("%08u ", t[i]);
	}
	printf("\n");
	
	_mm512_storeu_epi32(t, i7);
	for (i = 0; i < 16; i++)
	{
		printf("%08u ", t[i]);
	}
	printf("\n");
	
	_mm512_storeu_epi32(t, i8);
	for (i = 0; i < 16; i++)
	{
		printf("%08u ", t[i]);
	}
	printf("\n");
}


void bitonic_merge32_dir_128(uint32_t* data, int dir) 
{
	// perform the final merge phase on 128 32-bit elements
	int i, j;

	__m512i t1;
	__m512i t2;
	__m512i t3;
	__m512i t4;
	__m512i t5;
	__m512i t6;
	__m512i t7;
	__m512i t8;
	__m512i dv1;
	__m512i dv2;
	__m512i dv3;
	__m512i dv4;
	__m512i dv5;
	__m512i dv6;
	__m512i dv7;
	__m512i dv8;
	__m512i dv1_swap;
	__m512i dv2_swap;
	__m512i dv3_swap;
	__m512i dv4_swap;
	__m512i dv5_swap;
	__m512i dv6_swap;
	__m512i dv7_swap;
	__m512i dv8_swap;
	__mmask16 m1;
	__mmask16 m2;
	__mmask16 m3;
	__mmask16 m4;
	__mmask16 m5;
	__mmask16 m6;
	__mmask16 m7;
	__mmask16 m8;

	dv1 = _mm512_load_epi32(data);
	dv2 = _mm512_load_epi32(data + 16);
	dv3 = _mm512_load_epi32(data + 32);
	dv4 = _mm512_load_epi32(data + 48);
	dv5 = _mm512_load_epi32(data + 64);
	dv6 = _mm512_load_epi32(data + 80);
	dv7 = _mm512_load_epi32(data + 96);
	dv8 = _mm512_load_epi32(data + 112);

	if (dir == 1)
	{
		// distance 64 swaps - just compare the vecs
		m1 = _mm512_cmplt_epu32_mask(dv1, dv5);
		m2 = _mm512_cmplt_epu32_mask(dv2, dv6);
		m3 = _mm512_cmplt_epu32_mask(dv3, dv7);
		m4 = _mm512_cmplt_epu32_mask(dv4, dv8);

		t1 = _mm512_mask_blend_epi32(m1, dv1, dv5);
		t2 = _mm512_mask_blend_epi32(m2, dv2, dv6);
		t3 = _mm512_mask_blend_epi32(m3, dv3, dv7);
		t4 = _mm512_mask_blend_epi32(m4, dv4, dv8);
		dv5 = _mm512_mask_blend_epi32(m1, dv5, dv1);
		dv6 = _mm512_mask_blend_epi32(m2, dv6, dv2);
		dv7 = _mm512_mask_blend_epi32(m3, dv7, dv3);
		dv8 = _mm512_mask_blend_epi32(m4, dv8, dv4);
		dv1 = t1;
		dv2 = t2;
		dv3 = t3;
		dv4 = t4;
		
		// distance 32 swaps - just compare the vecs
		m1 = _mm512_cmplt_epu32_mask(dv1, dv3);
		m2 = _mm512_cmplt_epu32_mask(dv2, dv4);
		m3 = _mm512_cmplt_epu32_mask(dv5, dv7);
		m4 = _mm512_cmplt_epu32_mask(dv6, dv8);

		t1 = _mm512_mask_blend_epi32(m1, dv1, dv3);
		t2 = _mm512_mask_blend_epi32(m2, dv2, dv4);
		t3 = _mm512_mask_blend_epi32(m3, dv5, dv7);
		t4 = _mm512_mask_blend_epi32(m4, dv6, dv8);
			
		dv3 = _mm512_mask_blend_epi32(m1, dv3, dv1);
		dv4 = _mm512_mask_blend_epi32(m2, dv4, dv2);
		dv7 = _mm512_mask_blend_epi32(m3, dv7, dv5);
		dv8 = _mm512_mask_blend_epi32(m4, dv8, dv6);
		dv1 = t1;
		dv2 = t2;
		dv5 = t3;
		dv6 = t4;
		
		// distance 16 swaps - just compare the vecs
		m1 = _mm512_cmplt_epu32_mask(dv1, dv2);
		m2 = _mm512_cmplt_epu32_mask(dv3, dv4);
		m3 = _mm512_cmplt_epu32_mask(dv5, dv6);
		m4 = _mm512_cmplt_epu32_mask(dv7, dv8);

		t1 = _mm512_mask_blend_epi32(m1, dv1, dv2);
		t2 = _mm512_mask_blend_epi32(m2, dv3, dv4);
		t3 = _mm512_mask_blend_epi32(m3, dv5, dv6);
		t4 = _mm512_mask_blend_epi32(m4, dv7, dv8);
			
		dv2 = _mm512_mask_blend_epi32(m1, dv2, dv1);
		dv4 = _mm512_mask_blend_epi32(m2, dv4, dv3);
		dv6 = _mm512_mask_blend_epi32(m3, dv6, dv5);
		dv8 = _mm512_mask_blend_epi32(m4, dv8, dv7);
		dv1 = t1;
		dv3 = t2;
		dv5 = t3;
		dv7 = t4;

		// distance 8 swaps - 256 bits
		dv1_swap = SWAP256(dv1);
		dv2_swap = SWAP256(dv2);
		dv3_swap = SWAP256(dv3);
		dv4_swap = SWAP256(dv4);
		dv5_swap = SWAP256(dv5);
		dv6_swap = SWAP256(dv6);
		dv7_swap = SWAP256(dv7);
		dv8_swap = SWAP256(dv8);
		
		m1 = _mm512_cmplt_epu32_mask(dv1, dv1_swap);
		m2 = _mm512_cmplt_epu32_mask(dv2, dv2_swap);
		m3 = _mm512_cmplt_epu32_mask(dv3, dv3_swap);
		m4 = _mm512_cmplt_epu32_mask(dv4, dv4_swap);
		m5 = _mm512_cmplt_epu32_mask(dv5, dv5_swap);
		m6 = _mm512_cmplt_epu32_mask(dv6, dv6_swap);
		m7 = _mm512_cmplt_epu32_mask(dv7, dv7_swap);
		m8 = _mm512_cmplt_epu32_mask(dv8, dv8_swap);

		// 'FF00' for the swap between dist-8 lanes
		dv1 = _mm512_mask_blend_epi32(m1 ^ 0xFF00, dv1, dv1_swap);
		dv2 = _mm512_mask_blend_epi32(m2 ^ 0xFF00, dv2, dv2_swap);
		dv3 = _mm512_mask_blend_epi32(m3 ^ 0xFF00, dv3, dv3_swap);
		dv4 = _mm512_mask_blend_epi32(m4 ^ 0xFF00, dv4, dv4_swap);
		dv5 = _mm512_mask_blend_epi32(m5 ^ 0xFF00, dv5, dv5_swap);
		dv6 = _mm512_mask_blend_epi32(m6 ^ 0xFF00, dv6, dv6_swap);
		dv7 = _mm512_mask_blend_epi32(m7 ^ 0xFF00, dv7, dv7_swap);
		dv8 = _mm512_mask_blend_epi32(m8 ^ 0xFF00, dv8, dv8_swap);

		// distance 4 swaps
		dv1_swap = SWAP128(dv1);
		dv2_swap = SWAP128(dv2);
		dv3_swap = SWAP128(dv3);
		dv4_swap = SWAP128(dv4);
		dv5_swap = SWAP128(dv5);
		dv6_swap = SWAP128(dv6);
		dv7_swap = SWAP128(dv7);
		dv8_swap = SWAP128(dv8);
		
		m1 = _mm512_cmplt_epu32_mask(dv1, dv1_swap);
		m2 = _mm512_cmplt_epu32_mask(dv2, dv2_swap);
		m3 = _mm512_cmplt_epu32_mask(dv3, dv3_swap);
		m4 = _mm512_cmplt_epu32_mask(dv4, dv4_swap);
		m5 = _mm512_cmplt_epu32_mask(dv5, dv5_swap);
		m6 = _mm512_cmplt_epu32_mask(dv6, dv6_swap);
		m7 = _mm512_cmplt_epu32_mask(dv7, dv7_swap);
		m8 = _mm512_cmplt_epu32_mask(dv8, dv8_swap);

		// 'F0' for the swap between dist-4 lanes, non alternating
		dv1 = _mm512_mask_blend_epi32(m1 ^ 0xF0F0, dv1, dv1_swap);
		dv2 = _mm512_mask_blend_epi32(m2 ^ 0xF0F0, dv2, dv2_swap);
		dv3 = _mm512_mask_blend_epi32(m3 ^ 0xF0F0, dv3, dv3_swap);
		dv4 = _mm512_mask_blend_epi32(m4 ^ 0xF0F0, dv4, dv4_swap);
		dv5 = _mm512_mask_blend_epi32(m5 ^ 0xF0F0, dv5, dv5_swap);
		dv6 = _mm512_mask_blend_epi32(m6 ^ 0xF0F0, dv6, dv6_swap);
		dv7 = _mm512_mask_blend_epi32(m7 ^ 0xF0F0, dv7, dv7_swap);
		dv8 = _mm512_mask_blend_epi32(m8 ^ 0xF0F0, dv8, dv8_swap);

		// distance 2 swaps
		dv1_swap = SWAP64(dv1);
		dv2_swap = SWAP64(dv2);
		dv3_swap = SWAP64(dv3);
		dv4_swap = SWAP64(dv4);
		dv5_swap = SWAP64(dv5);
		dv6_swap = SWAP64(dv6);
		dv7_swap = SWAP64(dv7);
		dv8_swap = SWAP64(dv8);
		
		m1 = _mm512_cmplt_epu32_mask(dv1, dv1_swap);
		m2 = _mm512_cmplt_epu32_mask(dv2, dv2_swap);
		m3 = _mm512_cmplt_epu32_mask(dv3, dv3_swap);
		m4 = _mm512_cmplt_epu32_mask(dv4, dv4_swap);
		m5 = _mm512_cmplt_epu32_mask(dv5, dv5_swap);
		m6 = _mm512_cmplt_epu32_mask(dv6, dv6_swap);
		m7 = _mm512_cmplt_epu32_mask(dv7, dv7_swap);
		m8 = _mm512_cmplt_epu32_mask(dv8, dv8_swap);

		// 'CC' for the swap between dist-2 lanes, non alternating
		dv1 = _mm512_mask_blend_epi32(m1 ^ 0xCCCC, dv1, dv1_swap);
		dv2 = _mm512_mask_blend_epi32(m2 ^ 0xCCCC, dv2, dv2_swap);
		dv3 = _mm512_mask_blend_epi32(m3 ^ 0xCCCC, dv3, dv3_swap);
		dv4 = _mm512_mask_blend_epi32(m4 ^ 0xCCCC, dv4, dv4_swap);
		dv5 = _mm512_mask_blend_epi32(m5 ^ 0xCCCC, dv5, dv5_swap);
		dv6 = _mm512_mask_blend_epi32(m6 ^ 0xCCCC, dv6, dv6_swap);
		dv7 = _mm512_mask_blend_epi32(m7 ^ 0xCCCC, dv7, dv7_swap);
		dv8 = _mm512_mask_blend_epi32(m8 ^ 0xCCCC, dv8, dv8_swap);

		// adjacent swaps
		dv1_swap = SWAP32(dv1);
		dv2_swap = SWAP32(dv2);
		dv3_swap = SWAP32(dv3);
		dv4_swap = SWAP32(dv4);
		dv5_swap = SWAP32(dv5);
		dv6_swap = SWAP32(dv6);
		dv7_swap = SWAP32(dv7);
		dv8_swap = SWAP32(dv8);
		
		m1 = _mm512_cmplt_epu32_mask(dv1, dv1_swap);
		m2 = _mm512_cmplt_epu32_mask(dv2, dv2_swap);
		m3 = _mm512_cmplt_epu32_mask(dv3, dv3_swap);
		m4 = _mm512_cmplt_epu32_mask(dv4, dv4_swap);
		m5 = _mm512_cmplt_epu32_mask(dv5, dv5_swap);
		m6 = _mm512_cmplt_epu32_mask(dv6, dv6_swap);
		m7 = _mm512_cmplt_epu32_mask(dv7, dv7_swap);
		m8 = _mm512_cmplt_epu32_mask(dv8, dv8_swap);

		// 'AA' for the swap between adjacent lanes
		dv1 = _mm512_mask_blend_epi32(m1 ^ 0xAAAA, dv1, dv1_swap);
		dv2 = _mm512_mask_blend_epi32(m2 ^ 0xAAAA, dv2, dv2_swap);
		dv3 = _mm512_mask_blend_epi32(m3 ^ 0xAAAA, dv3, dv3_swap);
		dv4 = _mm512_mask_blend_epi32(m4 ^ 0xAAAA, dv4, dv4_swap);
		dv5 = _mm512_mask_blend_epi32(m5 ^ 0xAAAA, dv5, dv5_swap);
		dv6 = _mm512_mask_blend_epi32(m6 ^ 0xAAAA, dv6, dv6_swap);
		dv7 = _mm512_mask_blend_epi32(m7 ^ 0xAAAA, dv7, dv7_swap);
		dv8 = _mm512_mask_blend_epi32(m8 ^ 0xAAAA, dv8, dv8_swap);

	}
	else
	{

		// distance 64 swaps - just compare the vecs
		m1 = _mm512_cmpgt_epu32_mask(dv1, dv5);
		m2 = _mm512_cmpgt_epu32_mask(dv2, dv6);
		m3 = _mm512_cmpgt_epu32_mask(dv3, dv7);
		m4 = _mm512_cmpgt_epu32_mask(dv4, dv8);

		t1 = _mm512_mask_blend_epi32(m1, dv1, dv5);
		t2 = _mm512_mask_blend_epi32(m2, dv2, dv6);
		t3 = _mm512_mask_blend_epi32(m3, dv3, dv7);
		t4 = _mm512_mask_blend_epi32(m4, dv4, dv8);
		dv5 = _mm512_mask_blend_epi32(m1, dv5, dv1);
		dv6 = _mm512_mask_blend_epi32(m2, dv6, dv2);
		dv7 = _mm512_mask_blend_epi32(m3, dv7, dv3);
		dv8 = _mm512_mask_blend_epi32(m4, dv8, dv4);
		dv1 = t1;
		dv2 = t2;
		dv3 = t3;
		dv4 = t4;
		
		// distance 32 swaps - just compare the vecs
		m1 = _mm512_cmpgt_epu32_mask(dv1, dv3);
		m2 = _mm512_cmpgt_epu32_mask(dv2, dv4);
		m3 = _mm512_cmpgt_epu32_mask(dv5, dv7);
		m4 = _mm512_cmpgt_epu32_mask(dv6, dv8);

		t1 = _mm512_mask_blend_epi32(m1, dv1, dv3);
		t2 = _mm512_mask_blend_epi32(m2, dv2, dv4);
		t3 = _mm512_mask_blend_epi32(m3, dv5, dv7);
		t4 = _mm512_mask_blend_epi32(m4, dv6, dv8);
			
		dv3 = _mm512_mask_blend_epi32(m1, dv3, dv1);
		dv4 = _mm512_mask_blend_epi32(m2, dv4, dv2);
		dv7 = _mm512_mask_blend_epi32(m3, dv7, dv5);
		dv8 = _mm512_mask_blend_epi32(m4, dv8, dv6);
		dv1 = t1;
		dv2 = t2;
		dv5 = t3;
		dv6 = t4;
		
		// distance 16 swaps - just compare the vecs
		m1 = _mm512_cmpgt_epu32_mask(dv1, dv2);
		m2 = _mm512_cmpgt_epu32_mask(dv3, dv4);
		m3 = _mm512_cmpgt_epu32_mask(dv5, dv6);
		m4 = _mm512_cmpgt_epu32_mask(dv7, dv8);

		t1 = _mm512_mask_blend_epi32(m1, dv1, dv2);
		t2 = _mm512_mask_blend_epi32(m2, dv3, dv4);
		t3 = _mm512_mask_blend_epi32(m3, dv5, dv6);
		t4 = _mm512_mask_blend_epi32(m4, dv7, dv8);
			
		dv2 = _mm512_mask_blend_epi32(m1, dv2, dv1);
		dv4 = _mm512_mask_blend_epi32(m2, dv4, dv3);
		dv6 = _mm512_mask_blend_epi32(m3, dv6, dv5);
		dv8 = _mm512_mask_blend_epi32(m4, dv8, dv7);
		dv1 = t1;
		dv3 = t2;
		dv5 = t3;
		dv7 = t4;

		// distance 8 swaps - 256 bits
		dv1_swap = SWAP256(dv1);
		dv2_swap = SWAP256(dv2);
		dv3_swap = SWAP256(dv3);
		dv4_swap = SWAP256(dv4);
		dv5_swap = SWAP256(dv5);
		dv6_swap = SWAP256(dv6);
		dv7_swap = SWAP256(dv7);
		dv8_swap = SWAP256(dv8);
		
		m1 = _mm512_cmpgt_epu32_mask(dv1, dv1_swap);
		m2 = _mm512_cmpgt_epu32_mask(dv2, dv2_swap);
		m3 = _mm512_cmpgt_epu32_mask(dv3, dv3_swap);
		m4 = _mm512_cmpgt_epu32_mask(dv4, dv4_swap);
		m5 = _mm512_cmpgt_epu32_mask(dv5, dv5_swap);
		m6 = _mm512_cmpgt_epu32_mask(dv6, dv6_swap);
		m7 = _mm512_cmpgt_epu32_mask(dv7, dv7_swap);
		m8 = _mm512_cmpgt_epu32_mask(dv8, dv8_swap);

		// 'FF00' for the swap between dist-8 lanes
		dv1 = _mm512_mask_blend_epi32(m1 ^ 0xFF00, dv1, dv1_swap);
		dv2 = _mm512_mask_blend_epi32(m2 ^ 0xFF00, dv2, dv2_swap);
		dv3 = _mm512_mask_blend_epi32(m3 ^ 0xFF00, dv3, dv3_swap);
		dv4 = _mm512_mask_blend_epi32(m4 ^ 0xFF00, dv4, dv4_swap);
		dv5 = _mm512_mask_blend_epi32(m5 ^ 0xFF00, dv5, dv5_swap);
		dv6 = _mm512_mask_blend_epi32(m6 ^ 0xFF00, dv6, dv6_swap);
		dv7 = _mm512_mask_blend_epi32(m7 ^ 0xFF00, dv7, dv7_swap);
		dv8 = _mm512_mask_blend_epi32(m8 ^ 0xFF00, dv8, dv8_swap);

		// distance 4 swaps
		dv1_swap = SWAP128(dv1);
		dv2_swap = SWAP128(dv2);
		dv3_swap = SWAP128(dv3);
		dv4_swap = SWAP128(dv4);
		dv5_swap = SWAP128(dv5);
		dv6_swap = SWAP128(dv6);
		dv7_swap = SWAP128(dv7);
		dv8_swap = SWAP128(dv8);
		
		m1 = _mm512_cmpgt_epu32_mask(dv1, dv1_swap);
		m2 = _mm512_cmpgt_epu32_mask(dv2, dv2_swap);
		m3 = _mm512_cmpgt_epu32_mask(dv3, dv3_swap);
		m4 = _mm512_cmpgt_epu32_mask(dv4, dv4_swap);
		m5 = _mm512_cmpgt_epu32_mask(dv5, dv5_swap);
		m6 = _mm512_cmpgt_epu32_mask(dv6, dv6_swap);
		m7 = _mm512_cmpgt_epu32_mask(dv7, dv7_swap);
		m8 = _mm512_cmpgt_epu32_mask(dv8, dv8_swap);

		// 'F0' for the swap between dist-4 lanes, non alternating
		dv1 = _mm512_mask_blend_epi32(m1 ^ 0xF0F0, dv1, dv1_swap);
		dv2 = _mm512_mask_blend_epi32(m2 ^ 0xF0F0, dv2, dv2_swap);
		dv3 = _mm512_mask_blend_epi32(m3 ^ 0xF0F0, dv3, dv3_swap);
		dv4 = _mm512_mask_blend_epi32(m4 ^ 0xF0F0, dv4, dv4_swap);
		dv5 = _mm512_mask_blend_epi32(m5 ^ 0xF0F0, dv5, dv5_swap);
		dv6 = _mm512_mask_blend_epi32(m6 ^ 0xF0F0, dv6, dv6_swap);
		dv7 = _mm512_mask_blend_epi32(m7 ^ 0xF0F0, dv7, dv7_swap);
		dv8 = _mm512_mask_blend_epi32(m8 ^ 0xF0F0, dv8, dv8_swap);

		// distance 2 swaps
		dv1_swap = SWAP64(dv1);
		dv2_swap = SWAP64(dv2);
		dv3_swap = SWAP64(dv3);
		dv4_swap = SWAP64(dv4);
		dv5_swap = SWAP64(dv5);
		dv6_swap = SWAP64(dv6);
		dv7_swap = SWAP64(dv7);
		dv8_swap = SWAP64(dv8);
		
		m1 = _mm512_cmpgt_epu32_mask(dv1, dv1_swap);
		m2 = _mm512_cmpgt_epu32_mask(dv2, dv2_swap);
		m3 = _mm512_cmpgt_epu32_mask(dv3, dv3_swap);
		m4 = _mm512_cmpgt_epu32_mask(dv4, dv4_swap);
		m5 = _mm512_cmpgt_epu32_mask(dv5, dv5_swap);
		m6 = _mm512_cmpgt_epu32_mask(dv6, dv6_swap);
		m7 = _mm512_cmpgt_epu32_mask(dv7, dv7_swap);
		m8 = _mm512_cmpgt_epu32_mask(dv8, dv8_swap);

		// 'CC' for the swap between dist-2 lanes, non alternating
		dv1 = _mm512_mask_blend_epi32(m1 ^ 0xCCCC, dv1, dv1_swap);
		dv2 = _mm512_mask_blend_epi32(m2 ^ 0xCCCC, dv2, dv2_swap);
		dv3 = _mm512_mask_blend_epi32(m3 ^ 0xCCCC, dv3, dv3_swap);
		dv4 = _mm512_mask_blend_epi32(m4 ^ 0xCCCC, dv4, dv4_swap);
		dv5 = _mm512_mask_blend_epi32(m5 ^ 0xCCCC, dv5, dv5_swap);
		dv6 = _mm512_mask_blend_epi32(m6 ^ 0xCCCC, dv6, dv6_swap);
		dv7 = _mm512_mask_blend_epi32(m7 ^ 0xCCCC, dv7, dv7_swap);
		dv8 = _mm512_mask_blend_epi32(m8 ^ 0xCCCC, dv8, dv8_swap);

		// adjacent swaps
		dv1_swap = SWAP32(dv1);
		dv2_swap = SWAP32(dv2);
		dv3_swap = SWAP32(dv3);
		dv4_swap = SWAP32(dv4);
		dv5_swap = SWAP32(dv5);
		dv6_swap = SWAP32(dv6);
		dv7_swap = SWAP32(dv7);
		dv8_swap = SWAP32(dv8);
		
		m1 = _mm512_cmpgt_epu32_mask(dv1, dv1_swap);
		m2 = _mm512_cmpgt_epu32_mask(dv2, dv2_swap);
		m3 = _mm512_cmpgt_epu32_mask(dv3, dv3_swap);
		m4 = _mm512_cmpgt_epu32_mask(dv4, dv4_swap);
		m5 = _mm512_cmpgt_epu32_mask(dv5, dv5_swap);
		m6 = _mm512_cmpgt_epu32_mask(dv6, dv6_swap);
		m7 = _mm512_cmpgt_epu32_mask(dv7, dv7_swap);
		m8 = _mm512_cmpgt_epu32_mask(dv8, dv8_swap);

		// 'AA' for the swap between adjacent lanes
		dv1 = _mm512_mask_blend_epi32(m1 ^ 0xAAAA, dv1, dv1_swap);
		dv2 = _mm512_mask_blend_epi32(m2 ^ 0xAAAA, dv2, dv2_swap);
		dv3 = _mm512_mask_blend_epi32(m3 ^ 0xAAAA, dv3, dv3_swap);
		dv4 = _mm512_mask_blend_epi32(m4 ^ 0xAAAA, dv4, dv4_swap);
		dv5 = _mm512_mask_blend_epi32(m5 ^ 0xAAAA, dv5, dv5_swap);
		dv6 = _mm512_mask_blend_epi32(m6 ^ 0xAAAA, dv6, dv6_swap);
		dv7 = _mm512_mask_blend_epi32(m7 ^ 0xAAAA, dv7, dv7_swap);
		dv8 = _mm512_mask_blend_epi32(m8 ^ 0xAAAA, dv8, dv8_swap);
	}

	_mm512_store_epi32(data, dv1);
	_mm512_store_epi32(data + 16, dv2);
	_mm512_store_epi32(data + 32, dv3);
	_mm512_store_epi32(data + 48, dv4);
	_mm512_store_epi32(data + 64, dv5);
	_mm512_store_epi32(data + 80, dv6);
	_mm512_store_epi32(data + 96, dv7);
	_mm512_store_epi32(data + 112, dv8);

	return;
}

void bitonic_sort32_dir_128(uint32_t* data, int dir) 
{
	// sort 128 32-bit elements
	int i, j;

	__m512i t1;
	__m512i t2;
	__m512i t3;
	__m512i t4;
	__m512i t5;
	__m512i t6;
	__m512i t7;
	__m512i t8;
	__m512i dv1;
	__m512i dv2;
	__m512i dv3;
	__m512i dv4;
	__m512i dv5;
	__m512i dv6;
	__m512i dv7;
	__m512i dv8;
	__m512i dv1_swap;
	__m512i dv2_swap;
	__m512i dv3_swap;
	__m512i dv4_swap;
	__m512i dv5_swap;
	__m512i dv6_swap;
	__m512i dv7_swap;
	__m512i dv8_swap;
	__mmask16 m1;
	__mmask16 m2;
	__mmask16 m3;
	__mmask16 m4;
	__mmask16 m5;
	__mmask16 m6;
	__mmask16 m7;
	__mmask16 m8;

	dv1 = _mm512_load_epi32(data);
	dv2 = _mm512_load_epi32(data + 16);
	dv3 = _mm512_load_epi32(data + 32);
	dv4 = _mm512_load_epi32(data + 48);
	dv5 = _mm512_load_epi32(data + 64);
	dv6 = _mm512_load_epi32(data + 80);
	dv7 = _mm512_load_epi32(data + 96);
	dv8 = _mm512_load_epi32(data + 112);

	// phase 0: dist-2 alternating compares ('CC')
	
	// adjacent swaps, alternating compares
	dv1_swap = SWAP32(dv1);
	dv2_swap = SWAP32(dv2);
	dv3_swap = SWAP32(dv3);
	dv4_swap = SWAP32(dv4);
	dv5_swap = SWAP32(dv5);
	dv6_swap = SWAP32(dv6);
	dv7_swap = SWAP32(dv7);
	dv8_swap = SWAP32(dv8);
	
	// 'AA' for the swap between adjacent lanes ^ 'CC' to alternate gt vs. le --> 0x66
	//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);				// latency 3 throughput 1	--> 8 clocks
	//BLENDMASK(0x6666, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);	// latency 1 throughput 1	--> 8 clocks
	MINMAX(0x6666, 0x9999, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8)
	
	// phase 1: dist-4 alternating compares ('F0')
	
	// distance 2 swaps
	dv1_swap = SWAP64(dv1);
	dv2_swap = SWAP64(dv2);
	dv3_swap = SWAP64(dv3);
	dv4_swap = SWAP64(dv4);
	dv5_swap = SWAP64(dv5);
	dv6_swap = SWAP64(dv6);
	dv7_swap = SWAP64(dv7);
	dv8_swap = SWAP64(dv8);

	// 'CC' for the swap between dist-2 lanes ^ 'F0' to alternate gt vs. le --> 0x3C
	//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	//BLENDMASK(0x3C3C, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX(0x3C3C, 0xC3C3, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8)

	// adjacent swaps
	dv1_swap = SWAP32(dv1);
	dv2_swap = SWAP32(dv2);
	dv3_swap = SWAP32(dv3);
	dv4_swap = SWAP32(dv4);
	dv5_swap = SWAP32(dv5);
	dv6_swap = SWAP32(dv6);
	dv7_swap = SWAP32(dv7);
	dv8_swap = SWAP32(dv8);

	// 'AA' for the swap between adjacent lanes ^ 'F0' to alternate gt vs. le --> 0x5A
	//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	//BLENDMASK(0x5A5A, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX(0x5A5A, 0xA5A5, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	// phase 2: dist-8 alternating compares ('FF00')

	// distance 4 swaps
	dv1_swap = SWAP128(dv1);
	dv2_swap = SWAP128(dv2);
	dv3_swap = SWAP128(dv3);
	dv4_swap = SWAP128(dv4);
	dv5_swap = SWAP128(dv5);
	dv6_swap = SWAP128(dv6);
	dv7_swap = SWAP128(dv7);
	dv8_swap = SWAP128(dv8);

	// 'F0F0' for the swap between dist-4 lanes ^ 'FF00' to alternate gt vs. le --> 0x0FF0
	//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	//BLENDMASK(0x0FF0, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX(0x0FF0, 0xf00f, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	// distance 2 swaps
	dv1_swap = SWAP64(dv1);
	dv2_swap = SWAP64(dv2);
	dv3_swap = SWAP64(dv3);
	dv4_swap = SWAP64(dv4);
	dv5_swap = SWAP64(dv5);
	dv6_swap = SWAP64(dv6);
	dv7_swap = SWAP64(dv7);
	dv8_swap = SWAP64(dv8);

	// 'CCCC' for the swap between dist-2 lanes ^ 'FF00' to alternate gt vs. le --> 0x33CC
	//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	//BLENDMASK(0x33CC, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX(0x33CC, 0xcc33, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	// adjacent swaps
	dv1_swap = SWAP32(dv1);
	dv2_swap = SWAP32(dv2);
	dv3_swap = SWAP32(dv3);
	dv4_swap = SWAP32(dv4);
	dv5_swap = SWAP32(dv5);
	dv6_swap = SWAP32(dv6);
	dv7_swap = SWAP32(dv7);
	dv8_swap = SWAP32(dv8);

	// 'AA' for the swap between adjacent lanes ^ 'FF00' to alternate gt vs. le --> 0x55AA
	//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	//BLENDMASK(0x55AA, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX(0x55AA, 0xaa55, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	// phase 2: dist-16 alternating compares (alternating gt/lt)
	
	// distance 8 swaps (256 bits)
	dv1_swap = SWAP256(dv1);
	dv2_swap = SWAP256(dv2);
	dv3_swap = SWAP256(dv3);
	dv4_swap = SWAP256(dv4);
	dv5_swap = SWAP256(dv5);
	dv6_swap = SWAP256(dv6);
	dv7_swap = SWAP256(dv7);
	dv8_swap = SWAP256(dv8);

	// 'FF00' for the swap between dist-8 lanes, non alternating 
	// (b/c we do separate gt/lt compares of 16 elements with dist-8 swaps)
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT);
	//BLENDMASK(0xFF00, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt1(0xFF00, 0x00ff, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	// distance 4 swaps
	dv1_swap = SWAP128(dv1);
	dv2_swap = SWAP128(dv2);
	dv3_swap = SWAP128(dv3);
	dv4_swap = SWAP128(dv4);
	dv5_swap = SWAP128(dv5);
	dv6_swap = SWAP128(dv6);
	dv7_swap = SWAP128(dv7);
	dv8_swap = SWAP128(dv8);

	// 'F0' for the swap between dist-4 lanes, non alternating
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT);
	//BLENDMASK(0xF0F0, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt1(0xF0F0, 0x0f0f, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	// distance 2 swaps
	dv1_swap = SWAP64(dv1);
	dv2_swap = SWAP64(dv2);
	dv3_swap = SWAP64(dv3);
	dv4_swap = SWAP64(dv4);
	dv5_swap = SWAP64(dv5);
	dv6_swap = SWAP64(dv6);
	dv7_swap = SWAP64(dv7);
	dv8_swap = SWAP64(dv8);

	// 'CC' for the swap between dist-2 lanes, non alternating
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT);
	//BLENDMASK(0xCCCC, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt1(0xCCCC, 0x3333, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	// adjacent swaps
	dv1_swap = SWAP32(dv1);
	dv2_swap = SWAP32(dv2);
	dv3_swap = SWAP32(dv3);
	dv4_swap = SWAP32(dv4);
	dv5_swap = SWAP32(dv5);
	dv6_swap = SWAP32(dv6);
	dv7_swap = SWAP32(dv7);
	dv8_swap = SWAP32(dv8);

	// 'AA' for the swap between adjacent lanes, non alternating
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT);
	//BLENDMASK(0xAAAA, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt1(0xAAAA, 0x5555, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	// phase 3: dist-32 alternating compares (alternating gtgt/ltlt)

	// distance 16 swaps - just compare the vecs
	t1 =  _mm512_min_epu32(dv1, dv2);
	t2 =  _mm512_max_epu32(dv3, dv4);
	t3 =  _mm512_min_epu32(dv5, dv6);
	t4 =  _mm512_max_epu32(dv7, dv8);
	dv2 = _mm512_max_epu32(dv1, dv2);
	dv4 = _mm512_min_epu32(dv3, dv4);
	dv6 = _mm512_max_epu32(dv5, dv6);
	dv8 = _mm512_min_epu32(dv7, dv8);
	dv1 = t1;
	dv3 = t2;
	dv5 = t3;
	dv7 = t4;

	// distance 8 swaps (256 bits)
	dv1_swap = SWAP256(dv1);
	dv2_swap = SWAP256(dv2);
	dv3_swap = SWAP256(dv3);
	dv4_swap = SWAP256(dv4);
	dv5_swap = SWAP256(dv5);
	dv6_swap = SWAP256(dv6);
	dv7_swap = SWAP256(dv7);
	dv8_swap = SWAP256(dv8);

	// 'FF00' for the swap between dist-8 lanes, non alternating 
	// (b/c we do separate gt/lt compares of 16 elements with dist-8 swaps)
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xFF00, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt2(0xFF00, 0x00ff, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	// distance 4 swaps
	dv1_swap = SWAP128(dv1);
	dv2_swap = SWAP128(dv2);
	dv3_swap = SWAP128(dv3);
	dv4_swap = SWAP128(dv4);
	dv5_swap = SWAP128(dv5);
	dv6_swap = SWAP128(dv6);
	dv7_swap = SWAP128(dv7);
	dv8_swap = SWAP128(dv8);

	// 'F0' for the swap between dist-4 lanes, non alternating
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xF0F0, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt2(0xF0F0, 0x0f0f, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	// distance 2 swaps
	dv1_swap = SWAP64(dv1);
	dv2_swap = SWAP64(dv2);
	dv3_swap = SWAP64(dv3);
	dv4_swap = SWAP64(dv4);
	dv5_swap = SWAP64(dv5);
	dv6_swap = SWAP64(dv6);
	dv7_swap = SWAP64(dv7);
	dv8_swap = SWAP64(dv8);

	// 'CC' for the swap between dist-2 lanes, non alternating
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xCCCC, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt2(0xCCCC, 0x3333, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	// adjacent swaps
	dv1_swap = SWAP32(dv1);
	dv2_swap = SWAP32(dv2);
	dv3_swap = SWAP32(dv3);
	dv4_swap = SWAP32(dv4);
	dv5_swap = SWAP32(dv5);
	dv6_swap = SWAP32(dv6);
	dv7_swap = SWAP32(dv7);
	dv8_swap = SWAP32(dv8);

	// 'AA' for the swap between adjacent lanes
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xAAAA, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt2(0xAAAA, 0x5555, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	// phase 4: dist-64 alternating compares (alternating gtgtgtgt/ltltltlt)

	// distance 32 swaps - just compare the vecs
	t1 =  _mm512_min_epu32(dv1, dv3);
	t2 =  _mm512_min_epu32(dv2, dv4);
	t3 =  _mm512_max_epu32(dv5, dv7);
	t4 =  _mm512_max_epu32(dv6, dv8);
	dv3 = _mm512_max_epu32(dv1, dv3);
	dv4 = _mm512_max_epu32(dv2, dv4);
	dv7 = _mm512_min_epu32(dv5, dv7);
	dv8 = _mm512_min_epu32(dv6, dv8);
	dv1 = t1;
	dv2 = t2;
	dv5 = t3;
	dv6 = t4;
	
	// distance 16 swaps - just compare the vecs
	t1 =  _mm512_min_epu32(dv1, dv2);
	t2 =  _mm512_min_epu32(dv3, dv4);
	t3 =  _mm512_max_epu32(dv5, dv6);
	t4 =  _mm512_max_epu32(dv7, dv8);
	dv2 = _mm512_max_epu32(dv1, dv2);
	dv4 = _mm512_max_epu32(dv3, dv4);
	dv6 = _mm512_min_epu32(dv5, dv6);
	dv8 = _mm512_min_epu32(dv7, dv8);
	dv1 = t1;
	dv3 = t2;
	dv5 = t3;
	dv7 = t4;
	
	// distance 8 swaps (256 bits)
	dv1_swap = SWAP256(dv1);
	dv2_swap = SWAP256(dv2);
	dv3_swap = SWAP256(dv3);
	dv4_swap = SWAP256(dv4);
	dv5_swap = SWAP256(dv5);
	dv6_swap = SWAP256(dv6);
	dv7_swap = SWAP256(dv7);
	dv8_swap = SWAP256(dv8);

	// 'FF00' for the swap between dist-8 lanes, non alternating 
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, 
	//		_MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xFF00, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt4(0xFF00, 0x00ff, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	// distance 4 swaps
	dv1_swap = SWAP128(dv1);
	dv2_swap = SWAP128(dv2);
	dv3_swap = SWAP128(dv3);
	dv4_swap = SWAP128(dv4);
	dv5_swap = SWAP128(dv5);
	dv6_swap = SWAP128(dv6);
	dv7_swap = SWAP128(dv7);
	dv8_swap = SWAP128(dv8);

	// 'F0' for the swap between dist-4 lanes, non alternating
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, 
	//		_MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xF0F0, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt4(0xF0F0, 0x0f0f, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	// distance 2 swaps
	dv1_swap = SWAP64(dv1);
	dv2_swap = SWAP64(dv2);
	dv3_swap = SWAP64(dv3);
	dv4_swap = SWAP64(dv4);
	dv5_swap = SWAP64(dv5);
	dv6_swap = SWAP64(dv6);
	dv7_swap = SWAP64(dv7);
	dv8_swap = SWAP64(dv8);

	// 'CC' for the swap between dist-2 lanes, non alternating
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, 
	//		_MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xCCCC, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt4(0xCCCC, 0x3333, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	// adjacent swaps
	dv1_swap = SWAP32(dv1);
	dv2_swap = SWAP32(dv2);
	dv3_swap = SWAP32(dv3);
	dv4_swap = SWAP32(dv4);
	dv5_swap = SWAP32(dv5);
	dv6_swap = SWAP32(dv6);
	dv7_swap = SWAP32(dv7);
	dv8_swap = SWAP32(dv8);

	// 'AA' for the swap between adjacent lanes
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, 
	//		_MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xAAAA, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt4(0xAAAA, 0x5555, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);


	// phase 5: merge (all same compare 'dir')
	if (dir == 1)
	{
		// dist-64 swaps
		t1 =  _mm512_max_epu32(dv1, dv5);
		t2 =  _mm512_max_epu32(dv2, dv6);
		t3 =  _mm512_max_epu32(dv3, dv7);
		t4 =  _mm512_max_epu32(dv4, dv8);
		dv5 = _mm512_min_epu32(dv1, dv5);
		dv6 = _mm512_min_epu32(dv2, dv6);
		dv7 = _mm512_min_epu32(dv3, dv7);
		dv8 = _mm512_min_epu32(dv4, dv8);
		dv1 = t1;
		dv2 = t2;
		dv3 = t3;
		dv4 = t4;
		
		// distance 32 swaps - just compare the vecs
		t1 =  _mm512_max_epu32(dv1, dv3);
		t2 =  _mm512_max_epu32(dv2, dv4);
		t3 =  _mm512_max_epu32(dv5, dv7);
		t4 =  _mm512_max_epu32(dv6, dv8);
		dv3 = _mm512_min_epu32(dv1, dv3);
		dv4 = _mm512_min_epu32(dv2, dv4);
		dv7 = _mm512_min_epu32(dv5, dv7);
		dv8 = _mm512_min_epu32(dv6, dv8);
		dv1 = t1;
		dv2 = t2;
		dv5 = t3;
		dv6 = t4;

		// distance 16 swaps - just compare the vecs
		t1 =  _mm512_max_epu32(dv1, dv2);
		t2 =  _mm512_max_epu32(dv3, dv4);
		t3 =  _mm512_max_epu32(dv5, dv6);
		t4 =  _mm512_max_epu32(dv7, dv8);
		dv2 = _mm512_min_epu32(dv1, dv2);
		dv4 = _mm512_min_epu32(dv3, dv4);
		dv6 = _mm512_min_epu32(dv5, dv6);
		dv8 = _mm512_min_epu32(dv7, dv8);
		dv1 = t1;
		dv3 = t2;
		dv5 = t3;
		dv7 = t4;
		
		// distance 8 swaps - 256 bits
		dv1_swap = SWAP256(dv1);
		dv2_swap = SWAP256(dv2);
		dv3_swap = SWAP256(dv3);
		dv4_swap = SWAP256(dv4);
		dv5_swap = SWAP256(dv5);
		dv6_swap = SWAP256(dv6);
		dv7_swap = SWAP256(dv7);
		dv8_swap = SWAP256(dv8);

		// 'FF00' for the swap between dist-8 lanes
		//CMPLT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		//BLENDMASK(0xFF00, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		MINMAX(0x00ff, 0xFF00, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		
		// distance 4 swaps
		dv1_swap = SWAP128(dv1);
		dv2_swap = SWAP128(dv2);
		dv3_swap = SWAP128(dv3);
		dv4_swap = SWAP128(dv4);
		dv5_swap = SWAP128(dv5);
		dv6_swap = SWAP128(dv6);
		dv7_swap = SWAP128(dv7);
		dv8_swap = SWAP128(dv8);

		// 'F0' for the swap between dist-4 lanes, non alternating
		//CMPLT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		//BLENDMASK(0xF0F0, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		MINMAX(0x0f0f, 0xF0F0, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		
		// distance 2 swaps
		dv1_swap = SWAP64(dv1);
		dv2_swap = SWAP64(dv2);
		dv3_swap = SWAP64(dv3);
		dv4_swap = SWAP64(dv4);
		dv5_swap = SWAP64(dv5);
		dv6_swap = SWAP64(dv6);
		dv7_swap = SWAP64(dv7);
		dv8_swap = SWAP64(dv8);

		// 'CC' for the swap between dist-2 lanes, non alternating
		//CMPLT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		//BLENDMASK(0xCCCC, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		MINMAX(0x3333, 0xCCCC, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		
		// adjacent swaps
		dv1_swap = SWAP32(dv1);
		dv2_swap = SWAP32(dv2);
		dv3_swap = SWAP32(dv3);
		dv4_swap = SWAP32(dv4);
		dv5_swap = SWAP32(dv5);
		dv6_swap = SWAP32(dv6);
		dv7_swap = SWAP32(dv7);
		dv8_swap = SWAP32(dv8);

		// 'AA' for the swap between adjacent lanes
		//CMPLT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		//BLENDMASK(0xAAAA, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		MINMAX(0x5555, 0xaaaa, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	}
	else
	{	
		// dist-64 swaps
		t1 =  _mm512_min_epu32(dv1, dv5);
		t2 =  _mm512_min_epu32(dv2, dv6);
		t3 =  _mm512_min_epu32(dv3, dv7);
		t4 =  _mm512_min_epu32(dv4, dv8);
		dv5 = _mm512_max_epu32(dv1, dv5);
		dv6 = _mm512_max_epu32(dv2, dv6);
		dv7 = _mm512_max_epu32(dv3, dv7);
		dv8 = _mm512_max_epu32(dv4, dv8);
		dv1 = t1;
		dv2 = t2;
		dv3 = t3;
		dv4 = t4;
		
		// distance 32 swaps - just compare the vecs
		t1 =  _mm512_min_epu32(dv1, dv3);
		t2 =  _mm512_min_epu32(dv2, dv4);
		t3 =  _mm512_min_epu32(dv5, dv7);
		t4 =  _mm512_min_epu32(dv6, dv8);
		dv3 = _mm512_max_epu32(dv1, dv3);
		dv4 = _mm512_max_epu32(dv2, dv4);
		dv7 = _mm512_max_epu32(dv5, dv7);
		dv8 = _mm512_max_epu32(dv6, dv8);
		dv1 = t1;
		dv2 = t2;
		dv5 = t3;
		dv6 = t4;

		// distance 16 swaps - just compare the vecs
		t1 =  _mm512_min_epu32(dv1, dv2);
		t2 =  _mm512_min_epu32(dv3, dv4);
		t3 =  _mm512_min_epu32(dv5, dv6);
		t4 =  _mm512_min_epu32(dv7, dv8);
		dv2 = _mm512_max_epu32(dv1, dv2);
		dv4 = _mm512_max_epu32(dv3, dv4);
		dv6 = _mm512_max_epu32(dv5, dv6);
		dv8 = _mm512_max_epu32(dv7, dv8);
		dv1 = t1;
		dv3 = t2;
		dv5 = t3;
		dv7 = t4;

		// distance 8 swaps - 256 bits
		dv1_swap = SWAP256(dv1);
		dv2_swap = SWAP256(dv2);
		dv3_swap = SWAP256(dv3);
		dv4_swap = SWAP256(dv4);
		dv5_swap = SWAP256(dv5);
		dv6_swap = SWAP256(dv6);
		dv7_swap = SWAP256(dv7);
		dv8_swap = SWAP256(dv8);

		// '00FF' for the swap between dist-8 lanes
		//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		//BLENDMASK(0xFF00, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		MINMAX(0xFF00, 0x00ff, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		
		// distance 4 swaps
		dv1_swap = SWAP128(dv1);
		dv2_swap = SWAP128(dv2);
		dv3_swap = SWAP128(dv3);
		dv4_swap = SWAP128(dv4);
		dv5_swap = SWAP128(dv5);
		dv6_swap = SWAP128(dv6);
		dv7_swap = SWAP128(dv7);
		dv8_swap = SWAP128(dv8);

		// '0F' for the swap between dist-4 lanes, non alternating
		//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		//BLENDMASK(0xF0F0, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		MINMAX(0xf0f0, 0x0f0f, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

		// distance 2 swaps
		dv1_swap = SWAP64(dv1);
		dv2_swap = SWAP64(dv2);
		dv3_swap = SWAP64(dv3);
		dv4_swap = SWAP64(dv4);
		dv5_swap = SWAP64(dv5);
		dv6_swap = SWAP64(dv6);
		dv7_swap = SWAP64(dv7);
		dv8_swap = SWAP64(dv8);

		// 'CC' for the swap between dist-2 lanes, non alternating
		//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		//BLENDMASK(0xCCCC, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		MINMAX(0xCCCC, 0x3333, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		
		// adjacent swaps
		dv1_swap = SWAP32(dv1);
		dv2_swap = SWAP32(dv2);
		dv3_swap = SWAP32(dv3);
		dv4_swap = SWAP32(dv4);
		dv5_swap = SWAP32(dv5);
		dv6_swap = SWAP32(dv6);
		dv7_swap = SWAP32(dv7);
		dv8_swap = SWAP32(dv8);

		// 'AA' for the swap between adjacent lanes
		//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		//BLENDMASK(0xAAAA, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		MINMAX(0xaaaa, 0x5555, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	}

	_mm512_store_epi32(data, dv1);
	_mm512_store_epi32(data + 16, dv2);
	_mm512_store_epi32(data + 32, dv3);
	_mm512_store_epi32(data + 48, dv4);
	_mm512_store_epi32(data + 64, dv5);
	_mm512_store_epi32(data + 80, dv6);
	_mm512_store_epi32(data + 96, dv7);
	_mm512_store_epi32(data + 112, dv8);
	
	return;
}

void bitonic_sort32_dir_256(uint32_t* data, int dir) 
{
	// sort 256 32-bit elements
	int i, j;

	__m512i dv1;
	__m512i dv2;
	__m512i dv3;
	__m512i dv4;
	__m512i dv5;
	__m512i dv6;
	__m512i dv7;
	__m512i dv8;
	__m512i dv9;
	__m512i dv10;
	__m512i dv11;
	__m512i dv12;
	__m512i dv13;
	__m512i dv14;
	__m512i dv15;
	__m512i dv16;
	__m512i dv1_swap;
	__m512i dv2_swap;
	__m512i dv3_swap;
	__m512i dv4_swap;
	__m512i dv5_swap;
	__m512i dv6_swap;
	__m512i dv7_swap;
	__m512i dv8_swap;
	__m512i t1;
	__m512i t2;
	__m512i t3;
	__m512i t4;
	__m512i t5;
	__m512i t6;
	__m512i t7;
	__m512i t8;
	__mmask16 m1;
	__mmask16 m2;
	__mmask16 m3;
	__mmask16 m4;
	__mmask16 m5;
	__mmask16 m6;
	__mmask16 m7;
	__mmask16 m8;

	dv1 = _mm512_load_epi32(data);
	dv2 = _mm512_load_epi32(data + 16);
	dv3 = _mm512_load_epi32(data + 32);
	dv4 = _mm512_load_epi32(data + 48);
	dv5 = _mm512_load_epi32(data + 64);
	dv6 = _mm512_load_epi32(data + 80);
	dv7 = _mm512_load_epi32(data + 96);
	dv8 = _mm512_load_epi32(data + 112);
	dv9 = _mm512_load_epi32(data + 128);
	dv10 = _mm512_load_epi32(data + 9 * 16);
	dv11 = _mm512_load_epi32(data + 10 * 16);
	dv12 = _mm512_load_epi32(data + 11 * 16);
	dv13 = _mm512_load_epi32(data + 12 * 16);
	dv14 = _mm512_load_epi32(data + 13 * 16);
	dv15 = _mm512_load_epi32(data + 14 * 16);
	dv16 = _mm512_load_epi32(data + 15 * 16);

	// phase 0: dist-2 alternating compares ('CC')
	
	// adjacent swaps, alternating compares
	dv1_swap = SWAP32(dv1);
	dv2_swap = SWAP32(dv2);
	dv3_swap = SWAP32(dv3);
	dv4_swap = SWAP32(dv4);
	dv5_swap = SWAP32(dv5);
	dv6_swap = SWAP32(dv6);
	dv7_swap = SWAP32(dv7);
	dv8_swap = SWAP32(dv8);
	
	// 'AA' for the swap between adjacent lanes ^ 'CC' to alternate gt vs. le --> 0x66
	//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);				// latency 3 throughput 1	--> 8 clocks
	//BLENDMASK(0x6666, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);	// latency 1 throughput 1	--> 8 clocks
	MINMAX(0x6666, 0x9999, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8)

	
	dv1_swap = SWAP32(dv9);
	dv2_swap = SWAP32(dv10);
	dv3_swap = SWAP32(dv11);
	dv4_swap = SWAP32(dv12);
	dv5_swap = SWAP32(dv13);
	dv6_swap = SWAP32(dv14);
	dv7_swap = SWAP32(dv15);
	dv8_swap = SWAP32(dv16);
	
	//CMPGT(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	//BLENDMASK(0x6666, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	MINMAX(0x6666, 0x9999, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16)
	
	// phase 1: dist-4 alternating compares ('F0')
	
	// distance 2 swaps
	dv1_swap = SWAP64(dv1);
	dv2_swap = SWAP64(dv2);
	dv3_swap = SWAP64(dv3);
	dv4_swap = SWAP64(dv4);
	dv5_swap = SWAP64(dv5);
	dv6_swap = SWAP64(dv6);
	dv7_swap = SWAP64(dv7);
	dv8_swap = SWAP64(dv8);

	// 'CC' for the swap between dist-2 lanes ^ 'F0' to alternate gt vs. le --> 0x3C
	//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	//BLENDMASK(0x3C3C, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX(0x3C3C, 0xC3C3, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8)
	
	dv1_swap = SWAP64(dv9);
	dv2_swap = SWAP64(dv10);
	dv3_swap = SWAP64(dv11);
	dv4_swap = SWAP64(dv12);
	dv5_swap = SWAP64(dv13);
	dv6_swap = SWAP64(dv14);
	dv7_swap = SWAP64(dv15);
	dv8_swap = SWAP64(dv16);
	
	//CMPGT(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	//BLENDMASK(0x3C3C, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	MINMAX(0x3C3C, 0xC3C3, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);

	// adjacent swaps
	dv1_swap = SWAP32(dv1);
	dv2_swap = SWAP32(dv2);
	dv3_swap = SWAP32(dv3);
	dv4_swap = SWAP32(dv4);
	dv5_swap = SWAP32(dv5);
	dv6_swap = SWAP32(dv6);
	dv7_swap = SWAP32(dv7);
	dv8_swap = SWAP32(dv8);

	// 'AA' for the swap between adjacent lanes ^ 'F0' to alternate gt vs. le --> 0x5A
	//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	//BLENDMASK(0x5A5A, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX(0x5A5A, 0xA5A5, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	dv1_swap = SWAP32(dv9);
	dv2_swap = SWAP32(dv10);
	dv3_swap = SWAP32(dv11);
	dv4_swap = SWAP32(dv12);
	dv5_swap = SWAP32(dv13);
	dv6_swap = SWAP32(dv14);
	dv7_swap = SWAP32(dv15);
	dv8_swap = SWAP32(dv16);
	
	//CMPGT(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	//BLENDMASK(0x5a5a, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	MINMAX(0x5A5A, 0xA5A5, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);

	// phase 2: dist-8 alternating compares ('FF00')

	// distance 4 swaps
	dv1_swap = SWAP128(dv1);
	dv2_swap = SWAP128(dv2);
	dv3_swap = SWAP128(dv3);
	dv4_swap = SWAP128(dv4);
	dv5_swap = SWAP128(dv5);
	dv6_swap = SWAP128(dv6);
	dv7_swap = SWAP128(dv7);
	dv8_swap = SWAP128(dv8);

	// 'F0F0' for the swap between dist-4 lanes ^ 'FF00' to alternate gt vs. le --> 0x0FF0
	//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	//BLENDMASK(0x0FF0, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX(0x0FF0, 0xf00f, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	dv1_swap = SWAP128(dv9);
	dv2_swap = SWAP128(dv10);
	dv3_swap = SWAP128(dv11);
	dv4_swap = SWAP128(dv12);
	dv5_swap = SWAP128(dv13);
	dv6_swap = SWAP128(dv14);
	dv7_swap = SWAP128(dv15);
	dv8_swap = SWAP128(dv16);
	
	//CMPGT(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	//BLENDMASK(0x0ff0, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	MINMAX(0x0FF0, 0xf00f, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);

	// distance 2 swaps
	dv1_swap = SWAP64(dv1);
	dv2_swap = SWAP64(dv2);
	dv3_swap = SWAP64(dv3);
	dv4_swap = SWAP64(dv4);
	dv5_swap = SWAP64(dv5);
	dv6_swap = SWAP64(dv6);
	dv7_swap = SWAP64(dv7);
	dv8_swap = SWAP64(dv8);

	// 'CCCC' for the swap between dist-2 lanes ^ 'FF00' to alternate gt vs. le --> 0x33CC
	//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	//BLENDMASK(0x33CC, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX(0x33CC, 0xcc33, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	dv1_swap = SWAP64(dv9);
	dv2_swap = SWAP64(dv10);
	dv3_swap = SWAP64(dv11);
	dv4_swap = SWAP64(dv12);
	dv5_swap = SWAP64(dv13);
	dv6_swap = SWAP64(dv14);
	dv7_swap = SWAP64(dv15);
	dv8_swap = SWAP64(dv16);
	
	//CMPGT(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	//BLENDMASK(0x33cc, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	MINMAX(0x33CC, 0xcc33, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);

	// adjacent swaps
	dv1_swap = SWAP32(dv1);
	dv2_swap = SWAP32(dv2);
	dv3_swap = SWAP32(dv3);
	dv4_swap = SWAP32(dv4);
	dv5_swap = SWAP32(dv5);
	dv6_swap = SWAP32(dv6);
	dv7_swap = SWAP32(dv7);
	dv8_swap = SWAP32(dv8);

	// 'AA' for the swap between adjacent lanes ^ 'FF00' to alternate gt vs. le --> 0x55AA
	//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	//BLENDMASK(0x55AA, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX(0x55AA, 0xaa55, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	dv1_swap = SWAP32(dv9);
	dv2_swap = SWAP32(dv10);
	dv3_swap = SWAP32(dv11);
	dv4_swap = SWAP32(dv12);
	dv5_swap = SWAP32(dv13);
	dv6_swap = SWAP32(dv14);
	dv7_swap = SWAP32(dv15);
	dv8_swap = SWAP32(dv16);
	
	//CMPGT(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	//BLENDMASK(0x55aa, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	MINMAX(0x55AA, 0xaa55, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);

	// phase 2: dist-16 alternating compares (alternating gt/lt)
	
	// distance 8 swaps (256 bits)
	dv1_swap = SWAP256(dv1);
	dv2_swap = SWAP256(dv2);
	dv3_swap = SWAP256(dv3);
	dv4_swap = SWAP256(dv4);
	dv5_swap = SWAP256(dv5);
	dv6_swap = SWAP256(dv6);
	dv7_swap = SWAP256(dv7);
	dv8_swap = SWAP256(dv8);

	// 'FF00' for the swap between dist-8 lanes, non alternating 
	// (b/c we do separate gt/lt compares of 16 elements with dist-8 swaps)
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT);
	//BLENDMASK(0xFF00, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt1(0xFF00, 0x00ff, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	
	dv1_swap = SWAP256(dv9);
	dv2_swap = SWAP256(dv10);
	dv3_swap = SWAP256(dv11);
	dv4_swap = SWAP256(dv12);
	dv5_swap = SWAP256(dv13);
	dv6_swap = SWAP256(dv14);
	dv7_swap = SWAP256(dv15);
	dv8_swap = SWAP256(dv16);
	
	//CMP(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16,
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT);
	//BLENDMASK(0xff00, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	MINMAX_alt1(0xFF00, 0x00ff, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);

	// distance 4 swaps
	dv1_swap = SWAP128(dv1);
	dv2_swap = SWAP128(dv2);
	dv3_swap = SWAP128(dv3);
	dv4_swap = SWAP128(dv4);
	dv5_swap = SWAP128(dv5);
	dv6_swap = SWAP128(dv6);
	dv7_swap = SWAP128(dv7);
	dv8_swap = SWAP128(dv8);

	// 'F0' for the swap between dist-4 lanes, non alternating
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT);
	//BLENDMASK(0xF0F0, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt1(0xF0F0, 0x0f0f, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	dv1_swap = SWAP128(dv9);
	dv2_swap = SWAP128(dv10);
	dv3_swap = SWAP128(dv11);
	dv4_swap = SWAP128(dv12);
	dv5_swap = SWAP128(dv13);
	dv6_swap = SWAP128(dv14);
	dv7_swap = SWAP128(dv15);
	dv8_swap = SWAP128(dv16);
	
	//CMP(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16,
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT);
	//BLENDMASK(0xf0f0, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	MINMAX_alt1(0xF0F0, 0x0f0f, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);

	// distance 2 swaps
	dv1_swap = SWAP64(dv1);
	dv2_swap = SWAP64(dv2);
	dv3_swap = SWAP64(dv3);
	dv4_swap = SWAP64(dv4);
	dv5_swap = SWAP64(dv5);
	dv6_swap = SWAP64(dv6);
	dv7_swap = SWAP64(dv7);
	dv8_swap = SWAP64(dv8);

	// 'CC' for the swap between dist-2 lanes, non alternating
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT);
	//BLENDMASK(0xCCCC, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt1(0xCCCC, 0x3333, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	dv1_swap = SWAP64(dv9);
	dv2_swap = SWAP64(dv10);
	dv3_swap = SWAP64(dv11);
	dv4_swap = SWAP64(dv12);
	dv5_swap = SWAP64(dv13);
	dv6_swap = SWAP64(dv14);
	dv7_swap = SWAP64(dv15);
	dv8_swap = SWAP64(dv16);
	
	//CMP(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16,
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT);
	//BLENDMASK(0xcccc, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	MINMAX_alt1(0xCCCC, 0x3333, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);

	// adjacent swaps
	dv1_swap = SWAP32(dv1);
	dv2_swap = SWAP32(dv2);
	dv3_swap = SWAP32(dv3);
	dv4_swap = SWAP32(dv4);
	dv5_swap = SWAP32(dv5);
	dv6_swap = SWAP32(dv6);
	dv7_swap = SWAP32(dv7);
	dv8_swap = SWAP32(dv8);

	// 'AA' for the swap between adjacent lanes, non alternating
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT);
	//BLENDMASK(0xAAAA, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt1(0xAAAA, 0x5555, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	dv1_swap = SWAP32(dv9);
	dv2_swap = SWAP32(dv10);
	dv3_swap = SWAP32(dv11);
	dv4_swap = SWAP32(dv12);
	dv5_swap = SWAP32(dv13);
	dv6_swap = SWAP32(dv14);
	dv7_swap = SWAP32(dv15);
	dv8_swap = SWAP32(dv16);
	
	//CMP(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16,
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_NLE, _MM_CMPINT_LT);
	//BLENDMASK(0xaaaa, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	MINMAX_alt1(0xAAAA, 0x5555, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);

	// phase 3: dist-32 alternating compares (alternating gtgt/ltlt)

	// distance 16 swaps - just compare the vecs
	t1 =  _mm512_min_epu32(dv1, dv2);
	t2 =  _mm512_max_epu32(dv3, dv4);
	t3 =  _mm512_min_epu32(dv5, dv6);
	t4 =  _mm512_max_epu32(dv7, dv8);
	dv2 = _mm512_max_epu32(dv1, dv2);
	dv4 = _mm512_min_epu32(dv3, dv4);
	dv6 = _mm512_max_epu32(dv5, dv6);
	dv8 = _mm512_min_epu32(dv7, dv8);
	dv1 = t1;
	dv3 = t2;
	dv5 = t3;
	dv7 = t4;

	t1  =  _mm512_min_epu32(dv9 , dv10);
	t2  =  _mm512_max_epu32(dv11, dv12);
	t3  =  _mm512_min_epu32(dv13, dv14);
	t4  =  _mm512_max_epu32(dv15, dv16);
	dv10 = _mm512_max_epu32(dv9 , dv10);
	dv12 = _mm512_min_epu32(dv11, dv12);
	dv14 = _mm512_max_epu32(dv13, dv14);
	dv16 = _mm512_min_epu32(dv15, dv16);
	dv9  = t1;
	dv11 = t2;
	dv13 = t3;
	dv15 = t4;

	// distance 8 swaps (256 bits)
	dv1_swap = SWAP256(dv1);
	dv2_swap = SWAP256(dv2);
	dv3_swap = SWAP256(dv3);
	dv4_swap = SWAP256(dv4);
	dv5_swap = SWAP256(dv5);
	dv6_swap = SWAP256(dv6);
	dv7_swap = SWAP256(dv7);
	dv8_swap = SWAP256(dv8);

	// 'FF00' for the swap between dist-8 lanes, non alternating 
	// (b/c we do separate gt/lt compares of 16 elements with dist-8 swaps)
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xFF00, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt2(0xFF00, 0x00ff, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	dv1_swap = SWAP256(dv9);
	dv2_swap = SWAP256(dv10);
	dv3_swap = SWAP256(dv11);
	dv4_swap = SWAP256(dv12);
	dv5_swap = SWAP256(dv13);
	dv6_swap = SWAP256(dv14);
	dv7_swap = SWAP256(dv15);
	dv8_swap = SWAP256(dv16);
	
	//CMP(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xff00, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	MINMAX_alt2(0xFF00, 0x00ff, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);

	// distance 4 swaps
	dv1_swap = SWAP128(dv1);
	dv2_swap = SWAP128(dv2);
	dv3_swap = SWAP128(dv3);
	dv4_swap = SWAP128(dv4);
	dv5_swap = SWAP128(dv5);
	dv6_swap = SWAP128(dv6);
	dv7_swap = SWAP128(dv7);
	dv8_swap = SWAP128(dv8);

	// 'F0' for the swap between dist-4 lanes, non alternating
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xF0F0, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt2(0xF0F0, 0x0f0f, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	dv1_swap = SWAP128(dv9);
	dv2_swap = SWAP128(dv10);
	dv3_swap = SWAP128(dv11);
	dv4_swap = SWAP128(dv12);
	dv5_swap = SWAP128(dv13);
	dv6_swap = SWAP128(dv14);
	dv7_swap = SWAP128(dv15);
	dv8_swap = SWAP128(dv16);
	
	//CMP(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xf0f0, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	MINMAX_alt2(0xF0F0, 0x0f0f, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);

	// distance 2 swaps
	dv1_swap = SWAP64(dv1);
	dv2_swap = SWAP64(dv2);
	dv3_swap = SWAP64(dv3);
	dv4_swap = SWAP64(dv4);
	dv5_swap = SWAP64(dv5);
	dv6_swap = SWAP64(dv6);
	dv7_swap = SWAP64(dv7);
	dv8_swap = SWAP64(dv8);

	// 'CC' for the swap between dist-2 lanes, non alternating
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xCCCC, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt2(0xCCCC, 0x3333, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	dv1_swap = SWAP64(dv9);
	dv2_swap = SWAP64(dv10);
	dv3_swap = SWAP64(dv11);
	dv4_swap = SWAP64(dv12);
	dv5_swap = SWAP64(dv13);
	dv6_swap = SWAP64(dv14);
	dv7_swap = SWAP64(dv15);
	dv8_swap = SWAP64(dv16);
	
	//CMP(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xcccc, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	MINMAX_alt2(0xCCCC, 0x3333, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);

	// adjacent swaps
	dv1_swap = SWAP32(dv1);
	dv2_swap = SWAP32(dv2);
	dv3_swap = SWAP32(dv3);
	dv4_swap = SWAP32(dv4);
	dv5_swap = SWAP32(dv5);
	dv6_swap = SWAP32(dv6);
	dv7_swap = SWAP32(dv7);
	dv8_swap = SWAP32(dv8);

	// 'AA' for the swap between adjacent lanes
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xAAAA, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt2(0xAAAA, 0x5555, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	dv1_swap = SWAP32(dv9);
	dv2_swap = SWAP32(dv10);
	dv3_swap = SWAP32(dv11);
	dv4_swap = SWAP32(dv12);
	dv5_swap = SWAP32(dv13);
	dv6_swap = SWAP32(dv14);
	dv7_swap = SWAP32(dv15);
	dv8_swap = SWAP32(dv16);
	
	//CMP(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT, 
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xaaaa, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	MINMAX_alt2(0xAAAA, 0x5555, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	
	// phase 4: dist-64 alternating compares (alternating gtgtgtgt/ltltltlt)

	// distance 32 swaps - just compare the vecs
	t1 =  _mm512_min_epu32(dv1, dv3);
	t2 =  _mm512_min_epu32(dv2, dv4);
	t3 =  _mm512_max_epu32(dv5, dv7);
	t4 =  _mm512_max_epu32(dv6, dv8);
	dv3 = _mm512_max_epu32(dv1, dv3);
	dv4 = _mm512_max_epu32(dv2, dv4);
	dv7 = _mm512_min_epu32(dv5, dv7);
	dv8 = _mm512_min_epu32(dv6, dv8);
	dv1 = t1;
	dv2 = t2;
	dv5 = t3;
	dv6 = t4;

	t1  =  _mm512_min_epu32(dv9 , dv11);
	t2  =  _mm512_min_epu32(dv10, dv12);
	t3  =  _mm512_max_epu32(dv13, dv15);
	t4  =  _mm512_max_epu32(dv14, dv16);
	dv11 = _mm512_max_epu32(dv9 , dv11);
	dv12 = _mm512_max_epu32(dv10, dv12);
	dv15 = _mm512_min_epu32(dv13, dv15);
	dv16 = _mm512_min_epu32(dv14, dv16);
	dv9  = t1;
	dv10 = t2;
	dv13 = t3;
	dv14 = t4;
	
	// distance 16 swaps - just compare the vecs
	t1 =  _mm512_min_epu32(dv1, dv2);
	t2 =  _mm512_min_epu32(dv3, dv4);
	t3 =  _mm512_max_epu32(dv5, dv6);
	t4 =  _mm512_max_epu32(dv7, dv8);
	dv2 = _mm512_max_epu32(dv1, dv2);
	dv4 = _mm512_max_epu32(dv3, dv4);
	dv6 = _mm512_min_epu32(dv5, dv6);
	dv8 = _mm512_min_epu32(dv7, dv8);
	dv1 = t1;
	dv3 = t2;
	dv5 = t3;
	dv7 = t4;

	t1  =  _mm512_min_epu32(dv9 , dv10);
	t2  =  _mm512_min_epu32(dv11, dv12);
	t3  =  _mm512_max_epu32(dv13, dv14);
	t4  =  _mm512_max_epu32(dv15, dv16);
	dv10 = _mm512_max_epu32(dv9 , dv10);
	dv12 = _mm512_max_epu32(dv11, dv12);
	dv14 = _mm512_min_epu32(dv13, dv14);
	dv16 = _mm512_min_epu32(dv15, dv16);
	dv9  = t1;
	dv11 = t2;
	dv13 = t3;
	dv15 = t4;
	
	// distance 8 swaps (256 bits)
	dv1_swap = SWAP256(dv1);
	dv2_swap = SWAP256(dv2);
	dv3_swap = SWAP256(dv3);
	dv4_swap = SWAP256(dv4);
	dv5_swap = SWAP256(dv5);
	dv6_swap = SWAP256(dv6);
	dv7_swap = SWAP256(dv7);
	dv8_swap = SWAP256(dv8);

	// 'FF00' for the swap between dist-8 lanes, non alternating 
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, 
	//		_MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xFF00, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt4(0xFF00, 0x00ff, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	dv1_swap = SWAP256(dv9);
	dv2_swap = SWAP256(dv10);
	dv3_swap = SWAP256(dv11);
	dv4_swap = SWAP256(dv12);
	dv5_swap = SWAP256(dv13);
	dv6_swap = SWAP256(dv14);
	dv7_swap = SWAP256(dv15);
	dv8_swap = SWAP256(dv16);
	
	//CMP(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, 
	//		_MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xff00, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	MINMAX_alt4(0xFF00, 0x00ff, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	
	// distance 4 swaps
	dv1_swap = SWAP128(dv1);
	dv2_swap = SWAP128(dv2);
	dv3_swap = SWAP128(dv3);
	dv4_swap = SWAP128(dv4);
	dv5_swap = SWAP128(dv5);
	dv6_swap = SWAP128(dv6);
	dv7_swap = SWAP128(dv7);
	dv8_swap = SWAP128(dv8);

	// 'F0' for the swap between dist-4 lanes, non alternating
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, 
	//		_MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xF0F0, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt4(0xF0F0, 0x0f0f, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	dv1_swap = SWAP128(dv9);
	dv2_swap = SWAP128(dv10);
	dv3_swap = SWAP128(dv11);
	dv4_swap = SWAP128(dv12);
	dv5_swap = SWAP128(dv13);
	dv6_swap = SWAP128(dv14);
	dv7_swap = SWAP128(dv15);
	dv8_swap = SWAP128(dv16);
	
	//CMP(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, 
	//		_MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xf0f0, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	MINMAX_alt4(0xF0F0, 0x0f0f, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	
	// distance 2 swaps
	dv1_swap = SWAP64(dv1);
	dv2_swap = SWAP64(dv2);
	dv3_swap = SWAP64(dv3);
	dv4_swap = SWAP64(dv4);
	dv5_swap = SWAP64(dv5);
	dv6_swap = SWAP64(dv6);
	dv7_swap = SWAP64(dv7);
	dv8_swap = SWAP64(dv8);

	// 'CC' for the swap between dist-2 lanes, non alternating
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, 
	//		_MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xCCCC, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt4(0xCCCC, 0x3333, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	dv1_swap = SWAP64(dv9);
	dv2_swap = SWAP64(dv10);
	dv3_swap = SWAP64(dv11);
	dv4_swap = SWAP64(dv12);
	dv5_swap = SWAP64(dv13);
	dv6_swap = SWAP64(dv14);
	dv7_swap = SWAP64(dv15);
	dv8_swap = SWAP64(dv16);
	
	//CMP(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, 
	//		_MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xcccc, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	MINMAX_alt4(0xCCCC, 0x3333, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	
	// adjacent swaps
	dv1_swap = SWAP32(dv1);
	dv2_swap = SWAP32(dv2);
	dv3_swap = SWAP32(dv3);
	dv4_swap = SWAP32(dv4);
	dv5_swap = SWAP32(dv5);
	dv6_swap = SWAP32(dv6);
	dv7_swap = SWAP32(dv7);
	dv8_swap = SWAP32(dv8);

	// 'AA' for the swap between adjacent lanes
	//CMP(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, 
	//		_MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xAAAA, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX_alt4(0xAAAA, 0x5555, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	dv1_swap = SWAP32(dv9);
	dv2_swap = SWAP32(dv10);
	dv3_swap = SWAP32(dv11);
	dv4_swap = SWAP32(dv12);
	dv5_swap = SWAP32(dv13);
	dv6_swap = SWAP32(dv14);
	dv7_swap = SWAP32(dv15);
	dv8_swap = SWAP32(dv16);
	
	//CMP(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16,
	//		_MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, _MM_CMPINT_NLE, 
	//		_MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT, _MM_CMPINT_LT);
	//BLENDMASK(0xaaaa, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	MINMAX_alt4(0xAAAA, 0x5555, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	
	
	// phase 5: dist-128 alternating compares (alternating gtgtgtgtgtgtgtgt/ltltltltltltltlt)
	
	// dist-64 swaps
	t1 =  _mm512_min_epu32(dv1, dv5);
	t2 =  _mm512_min_epu32(dv2, dv6);
	t3 =  _mm512_min_epu32(dv3, dv7);
	t4 =  _mm512_min_epu32(dv4, dv8);
	dv5 = _mm512_max_epu32(dv1, dv5);
	dv6 = _mm512_max_epu32(dv2, dv6);
	dv7 = _mm512_max_epu32(dv3, dv7);
	dv8 = _mm512_max_epu32(dv4, dv8);
	dv1 = t1;
	dv2 = t2;
	dv3 = t3;
	dv4 = t4;

	t1  =  _mm512_max_epu32(dv9 , dv13);
	t2  =  _mm512_max_epu32(dv10, dv14);
	t3  =  _mm512_max_epu32(dv11, dv15);
	t4  =  _mm512_max_epu32(dv12, dv16);
	dv13 = _mm512_min_epu32(dv9 , dv13);
	dv14 = _mm512_min_epu32(dv10, dv14);
	dv15 = _mm512_min_epu32(dv11, dv15);
	dv16 = _mm512_min_epu32(dv12, dv16);
	dv9  = t1;
	dv10 = t2;
	dv11 = t3;
	dv12 = t4;
	
	
	// distance 32 swaps - just compare the vecs
	t1 =  _mm512_min_epu32(dv1, dv3);
	t2 =  _mm512_min_epu32(dv2, dv4);
	t3 =  _mm512_min_epu32(dv5, dv7);
	t4 =  _mm512_min_epu32(dv6, dv8);
	dv3 = _mm512_max_epu32(dv1, dv3);
	dv4 = _mm512_max_epu32(dv2, dv4);
	dv7 = _mm512_max_epu32(dv5, dv7);
	dv8 = _mm512_max_epu32(dv6, dv8);
	dv1 = t1;
	dv2 = t2;
	dv5 = t3;
	dv6 = t4;

	t1  =  _mm512_max_epu32(dv9 , dv11);
	t2  =  _mm512_max_epu32(dv10, dv12);
	t3  =  _mm512_max_epu32(dv13, dv15);
	t4  =  _mm512_max_epu32(dv14, dv16);
	dv11 = _mm512_min_epu32(dv9 , dv11);
	dv12 = _mm512_min_epu32(dv10, dv12);
	dv15 = _mm512_min_epu32(dv13, dv15);
	dv16 = _mm512_min_epu32(dv14, dv16);
	dv9  = t1;
	dv10 = t2;
	dv13 = t3;
	dv14 = t4;
	
	// distance 16 swaps - just compare the vecs
	t1 =  _mm512_min_epu32(dv1, dv2);
	t2 =  _mm512_min_epu32(dv3, dv4);
	t3 =  _mm512_min_epu32(dv5, dv6);
	t4 =  _mm512_min_epu32(dv7, dv8);
	dv2 = _mm512_max_epu32(dv1, dv2);
	dv4 = _mm512_max_epu32(dv3, dv4);
	dv6 = _mm512_max_epu32(dv5, dv6);
	dv8 = _mm512_max_epu32(dv7, dv8);
	dv1 = t1;
	dv3 = t2;
	dv5 = t3;
	dv7 = t4;

	t1  =  _mm512_max_epu32(dv9 , dv10);
	t2  =  _mm512_max_epu32(dv11, dv12);
	t3  =  _mm512_max_epu32(dv13, dv14);
	t4  =  _mm512_max_epu32(dv15, dv16);
	dv10 = _mm512_min_epu32(dv9 , dv10);
	dv12 = _mm512_min_epu32(dv11, dv12);
	dv14 = _mm512_min_epu32(dv13, dv14);
	dv16 = _mm512_min_epu32(dv15, dv16);
	dv9  = t1;
	dv11 = t2;
	dv13 = t3;
	dv15 = t4;
	
	// distance 8 swaps (256 bits)
	dv1_swap = SWAP256(dv1);
	dv2_swap = SWAP256(dv2);
	dv3_swap = SWAP256(dv3);
	dv4_swap = SWAP256(dv4);
	dv5_swap = SWAP256(dv5);
	dv6_swap = SWAP256(dv6);
	dv7_swap = SWAP256(dv7);
	dv8_swap = SWAP256(dv8);

	// 'FF00' for the swap between dist-8 lanes, non alternating 
	//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	//BLENDMASK(0xFF00, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX(0xff00, 0x00ff, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	dv1_swap = SWAP256(dv9);
	dv2_swap = SWAP256(dv10);
	dv3_swap = SWAP256(dv11);
	dv4_swap = SWAP256(dv12);
	dv5_swap = SWAP256(dv13);
	dv6_swap = SWAP256(dv14);
	dv7_swap = SWAP256(dv15);
	dv8_swap = SWAP256(dv16);
	
	//CMPLT(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	//BLENDMASK(0xff00, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	MINMAX(0x00ff, 0xff00, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	
	// distance 4 swaps
	dv1_swap = SWAP128(dv1);
	dv2_swap = SWAP128(dv2);
	dv3_swap = SWAP128(dv3);
	dv4_swap = SWAP128(dv4);
	dv5_swap = SWAP128(dv5);
	dv6_swap = SWAP128(dv6);
	dv7_swap = SWAP128(dv7);
	dv8_swap = SWAP128(dv8);

	// 'F0' for the swap between dist-4 lanes, non alternating
	//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	//BLENDMASK(0xF0F0, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX(0xF0F0, 0x0f0f, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	dv1_swap = SWAP128(dv9);
	dv2_swap = SWAP128(dv10);
	dv3_swap = SWAP128(dv11);
	dv4_swap = SWAP128(dv12);
	dv5_swap = SWAP128(dv13);
	dv6_swap = SWAP128(dv14);
	dv7_swap = SWAP128(dv15);
	dv8_swap = SWAP128(dv16);
	
	//CMPLT(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	//BLENDMASK(0xf0f0, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	MINMAX(0x0f0f, 0xF0F0, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	
	// distance 2 swaps
	dv1_swap = SWAP64(dv1);
	dv2_swap = SWAP64(dv2);
	dv3_swap = SWAP64(dv3);
	dv4_swap = SWAP64(dv4);
	dv5_swap = SWAP64(dv5);
	dv6_swap = SWAP64(dv6);
	dv7_swap = SWAP64(dv7);
	dv8_swap = SWAP64(dv8);

	// 'CC' for the swap between dist-2 lanes, non alternating
	//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	//BLENDMASK(0xCCCC, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX(0xcccc, 0x3333, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	dv1_swap = SWAP64(dv9);
	dv2_swap = SWAP64(dv10);
	dv3_swap = SWAP64(dv11);
	dv4_swap = SWAP64(dv12);
	dv5_swap = SWAP64(dv13);
	dv6_swap = SWAP64(dv14);
	dv7_swap = SWAP64(dv15);
	dv8_swap = SWAP64(dv16);
	
	//CMPLT(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	//BLENDMASK(0xcccc, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	MINMAX(0x3333, 0xcccc, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	
	// adjacent swaps
	dv1_swap = SWAP32(dv1);
	dv2_swap = SWAP32(dv2);
	dv3_swap = SWAP32(dv3);
	dv4_swap = SWAP32(dv4);
	dv5_swap = SWAP32(dv5);
	dv6_swap = SWAP32(dv6);
	dv7_swap = SWAP32(dv7);
	dv8_swap = SWAP32(dv8);

	// 'AA' for the swap between adjacent lanes
	//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	//BLENDMASK(0xAAAA, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	MINMAX(0xAAAA, 0x5555, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	
	dv1_swap = SWAP32(dv9);
	dv2_swap = SWAP32(dv10);
	dv3_swap = SWAP32(dv11);
	dv4_swap = SWAP32(dv12);
	dv5_swap = SWAP32(dv13);
	dv6_swap = SWAP32(dv14);
	dv7_swap = SWAP32(dv15);
	dv8_swap = SWAP32(dv16);
	
	//CMPLT(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	//BLENDMASK(0xaaaa, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	MINMAX(0x5555, 0xAAAA, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	

	// phase 6: merge (all same compare 'dir')
	if (dir == 1)
	{
		// distance 128 swaps - just compare the vecs
		t1 =   _mm512_max_epu32(dv1, dv9 );
		t2 =   _mm512_max_epu32(dv2, dv10);
		t3 =   _mm512_max_epu32(dv3, dv11);
		t4 =   _mm512_max_epu32(dv4, dv12);
		dv9  = _mm512_min_epu32(dv1, dv9 );
		dv10 = _mm512_min_epu32(dv2, dv10);
		dv11 = _mm512_min_epu32(dv3, dv11);
		dv12 = _mm512_min_epu32(dv4, dv12);
		dv1 = t1;
		dv2 = t2;
		dv3 = t3;
		dv4 = t4;

		t1  =  _mm512_max_epu32(dv5, dv13);
		t2  =  _mm512_max_epu32(dv6, dv14);
		t3  =  _mm512_max_epu32(dv7, dv15);
		t4  =  _mm512_max_epu32(dv8, dv16);
		dv13 = _mm512_min_epu32(dv5, dv13);
		dv14 = _mm512_min_epu32(dv6, dv14);
		dv15 = _mm512_min_epu32(dv7, dv15);
		dv16 = _mm512_min_epu32(dv8, dv16);
		dv5 = t1;
		dv6 = t2;
		dv7 = t3;
		dv8 = t4;
		
		// dist-64 swaps
		t1 =  _mm512_max_epu32(dv1, dv5);
		t2 =  _mm512_max_epu32(dv2, dv6);
		t3 =  _mm512_max_epu32(dv3, dv7);
		t4 =  _mm512_max_epu32(dv4, dv8);
		dv5 = _mm512_min_epu32(dv1, dv5);
		dv6 = _mm512_min_epu32(dv2, dv6);
		dv7 = _mm512_min_epu32(dv3, dv7);
		dv8 = _mm512_min_epu32(dv4, dv8);
		dv1 = t1;
		dv2 = t2;
		dv3 = t3;
		dv4 = t4;

		t1  =  _mm512_max_epu32(dv9 , dv13);
		t2  =  _mm512_max_epu32(dv10, dv14);
		t3  =  _mm512_max_epu32(dv11, dv15);
		t4  =  _mm512_max_epu32(dv12, dv16);
		dv13 = _mm512_min_epu32(dv9 , dv13);
		dv14 = _mm512_min_epu32(dv10, dv14);
		dv15 = _mm512_min_epu32(dv11, dv15);
		dv16 = _mm512_min_epu32(dv12, dv16);
		dv9  = t1;
		dv10 = t2;
		dv11 = t3;
		dv12 = t4;
		
		
		// distance 32 swaps - just compare the vecs
		t1 =  _mm512_max_epu32(dv1, dv3);
		t2 =  _mm512_max_epu32(dv2, dv4);
		t3 =  _mm512_max_epu32(dv5, dv7);
		t4 =  _mm512_max_epu32(dv6, dv8);
		dv3 = _mm512_min_epu32(dv1, dv3);
		dv4 = _mm512_min_epu32(dv2, dv4);
		dv7 = _mm512_min_epu32(dv5, dv7);
		dv8 = _mm512_min_epu32(dv6, dv8);
		dv1 = t1;
		dv2 = t2;
		dv5 = t3;
		dv6 = t4;

		t1  =  _mm512_max_epu32(dv9 , dv11);
		t2  =  _mm512_max_epu32(dv10, dv12);
		t3  =  _mm512_max_epu32(dv13, dv15);
		t4  =  _mm512_max_epu32(dv14, dv16);
		dv11 = _mm512_min_epu32(dv9 , dv11);
		dv12 = _mm512_min_epu32(dv10, dv12);
		dv15 = _mm512_min_epu32(dv13, dv15);
		dv16 = _mm512_min_epu32(dv14, dv16);
		dv9  = t1;
		dv10 = t2;
		dv13 = t3;
		dv14 = t4;
		
		// distance 16 swaps - just compare the vecs
		t1 =  _mm512_max_epu32(dv1, dv2);
		t2 =  _mm512_max_epu32(dv3, dv4);
		t3 =  _mm512_max_epu32(dv5, dv6);
		t4 =  _mm512_max_epu32(dv7, dv8);
		dv2 = _mm512_min_epu32(dv1, dv2);
		dv4 = _mm512_min_epu32(dv3, dv4);
		dv6 = _mm512_min_epu32(dv5, dv6);
		dv8 = _mm512_min_epu32(dv7, dv8);
		dv1 = t1;
		dv3 = t2;
		dv5 = t3;
		dv7 = t4;

		t1  =  _mm512_max_epu32(dv9 , dv10);
		t2  =  _mm512_max_epu32(dv11, dv12);
		t3  =  _mm512_max_epu32(dv13, dv14);
		t4  =  _mm512_max_epu32(dv15, dv16);
		dv10 = _mm512_min_epu32(dv9 , dv10);
		dv12 = _mm512_min_epu32(dv11, dv12);
		dv14 = _mm512_min_epu32(dv13, dv14);
		dv16 = _mm512_min_epu32(dv15, dv16);
		dv9  = t1;
		dv11 = t2;
		dv13 = t3;
		dv15 = t4;
		
		// distance 8 swaps - 256 bits
		dv1_swap = SWAP256(dv1);
		dv2_swap = SWAP256(dv2);
		dv3_swap = SWAP256(dv3);
		dv4_swap = SWAP256(dv4);
		dv5_swap = SWAP256(dv5);
		dv6_swap = SWAP256(dv6);
		dv7_swap = SWAP256(dv7);
		dv8_swap = SWAP256(dv8);

		// 'FF00' for the swap between dist-8 lanes
		//CMPLT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		//BLENDMASK(0xFF00, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		MINMAX(0x00ff, 0xFF00, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		
		dv1_swap = SWAP256(dv9);
		dv2_swap = SWAP256(dv10);
		dv3_swap = SWAP256(dv11);
		dv4_swap = SWAP256(dv12);
		dv5_swap = SWAP256(dv13);
		dv6_swap = SWAP256(dv14);
		dv7_swap = SWAP256(dv15);
		dv8_swap = SWAP256(dv16);
		
		//CMPLT(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
		//BLENDMASK(0xff00, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
		MINMAX(0x00ff, 0xFF00, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
		
		// distance 4 swaps
		dv1_swap = SWAP128(dv1);
		dv2_swap = SWAP128(dv2);
		dv3_swap = SWAP128(dv3);
		dv4_swap = SWAP128(dv4);
		dv5_swap = SWAP128(dv5);
		dv6_swap = SWAP128(dv6);
		dv7_swap = SWAP128(dv7);
		dv8_swap = SWAP128(dv8);

		// 'F0' for the swap between dist-4 lanes, non alternating
		//CMPLT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		//BLENDMASK(0xF0F0, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		MINMAX(0x0f0f, 0xF0F0, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		
		dv1_swap = SWAP128(dv9);
		dv2_swap = SWAP128(dv10);
		dv3_swap = SWAP128(dv11);
		dv4_swap = SWAP128(dv12);
		dv5_swap = SWAP128(dv13);
		dv6_swap = SWAP128(dv14);
		dv7_swap = SWAP128(dv15);
		dv8_swap = SWAP128(dv16);
		
		//CMPLT(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
		//BLENDMASK(0xf0f0, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
		MINMAX(0x0f0f, 0xF0F0, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
		
		// distance 2 swaps
		dv1_swap = SWAP64(dv1);
		dv2_swap = SWAP64(dv2);
		dv3_swap = SWAP64(dv3);
		dv4_swap = SWAP64(dv4);
		dv5_swap = SWAP64(dv5);
		dv6_swap = SWAP64(dv6);
		dv7_swap = SWAP64(dv7);
		dv8_swap = SWAP64(dv8);

		// 'CC' for the swap between dist-2 lanes, non alternating
		//CMPLT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		//BLENDMASK(0xCCCC, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		MINMAX(0x3333, 0xCCCC, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		
		dv1_swap = SWAP64(dv9);
		dv2_swap = SWAP64(dv10);
		dv3_swap = SWAP64(dv11);
		dv4_swap = SWAP64(dv12);
		dv5_swap = SWAP64(dv13);
		dv6_swap = SWAP64(dv14);
		dv7_swap = SWAP64(dv15);
		dv8_swap = SWAP64(dv16);
		
		//CMPLT(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
		//BLENDMASK(0xcccc, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
		MINMAX(0x3333, 0xCCCC, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
		
		// adjacent swaps
		dv1_swap = SWAP32(dv1);
		dv2_swap = SWAP32(dv2);
		dv3_swap = SWAP32(dv3);
		dv4_swap = SWAP32(dv4);
		dv5_swap = SWAP32(dv5);
		dv6_swap = SWAP32(dv6);
		dv7_swap = SWAP32(dv7);
		dv8_swap = SWAP32(dv8);

		// 'AA' for the swap between adjacent lanes
		//CMPLT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		//BLENDMASK(0xAAAA, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		MINMAX(0x5555, 0xaaaa, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		
		dv1_swap = SWAP32(dv9);
		dv2_swap = SWAP32(dv10);
		dv3_swap = SWAP32(dv11);
		dv4_swap = SWAP32(dv12);
		dv5_swap = SWAP32(dv13);
		dv6_swap = SWAP32(dv14);
		dv7_swap = SWAP32(dv15);
		dv8_swap = SWAP32(dv16);
		
		//CMPLT(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
		//BLENDMASK(0xaaaa, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
		MINMAX(0x5555, 0xaaaa, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
	}
	else
	{
		// distance 128 swaps - just compare the vecs
		t1 =  _mm512_min_epu32(dv1, dv9 );
		t2 =  _mm512_min_epu32(dv2, dv10);
		t3 =  _mm512_min_epu32(dv3, dv11);
		t4 =  _mm512_min_epu32(dv4, dv12);
		dv9  = _mm512_max_epu32(dv1, dv9 );
		dv10 = _mm512_max_epu32(dv2, dv10);
		dv11 = _mm512_max_epu32(dv3, dv11);
		dv12 = _mm512_max_epu32(dv4, dv12);
		dv1 = t1;
		dv2 = t2;
		dv3 = t3;
		dv4 = t4;

		t1  =  _mm512_min_epu32(dv5, dv13);
		t2  =  _mm512_min_epu32(dv6, dv14);
		t3  =  _mm512_min_epu32(dv7, dv15);
		t4  =  _mm512_min_epu32(dv8, dv16);
		dv13 = _mm512_max_epu32(dv5, dv13);
		dv14 = _mm512_max_epu32(dv6, dv14);
		dv15 = _mm512_max_epu32(dv7, dv15);
		dv16 = _mm512_max_epu32(dv8, dv16);
		dv5 = t1;
		dv6 = t2;
		dv7 = t3;
		dv8 = t4;
		
		// dist-64 swaps
		t1 =  _mm512_min_epu32(dv1, dv5);
		t2 =  _mm512_min_epu32(dv2, dv6);
		t3 =  _mm512_min_epu32(dv3, dv7);
		t4 =  _mm512_min_epu32(dv4, dv8);
		dv5 = _mm512_max_epu32(dv1, dv5);
		dv6 = _mm512_max_epu32(dv2, dv6);
		dv7 = _mm512_max_epu32(dv3, dv7);
		dv8 = _mm512_max_epu32(dv4, dv8);
		dv1 = t1;
		dv2 = t2;
		dv3 = t3;
		dv4 = t4;

		t1  =  _mm512_min_epu32(dv9 , dv13);
		t2  =  _mm512_min_epu32(dv10, dv14);
		t3  =  _mm512_min_epu32(dv11, dv15);
		t4  =  _mm512_min_epu32(dv12, dv16);
		dv13 = _mm512_max_epu32(dv9 , dv13);
		dv14 = _mm512_max_epu32(dv10, dv14);
		dv15 = _mm512_max_epu32(dv11, dv15);
		dv16 = _mm512_max_epu32(dv12, dv16);
		dv9  = t1;
		dv10 = t2;
		dv11 = t3;
		dv12 = t4;
		
		
		// distance 32 swaps - just compare the vecs
		t1 =  _mm512_min_epu32(dv1, dv3);
		t2 =  _mm512_min_epu32(dv2, dv4);
		t3 =  _mm512_min_epu32(dv5, dv7);
		t4 =  _mm512_min_epu32(dv6, dv8);
		dv3 = _mm512_max_epu32(dv1, dv3);
		dv4 = _mm512_max_epu32(dv2, dv4);
		dv7 = _mm512_max_epu32(dv5, dv7);
		dv8 = _mm512_max_epu32(dv6, dv8);
		dv1 = t1;
		dv2 = t2;
		dv5 = t3;
		dv6 = t4;

		t1  =  _mm512_min_epu32(dv9 , dv11);
		t2  =  _mm512_min_epu32(dv10, dv12);
		t3  =  _mm512_min_epu32(dv13, dv15);
		t4  =  _mm512_min_epu32(dv14, dv16);
		dv11 = _mm512_max_epu32(dv9 , dv11);
		dv12 = _mm512_max_epu32(dv10, dv12);
		dv15 = _mm512_max_epu32(dv13, dv15);
		dv16 = _mm512_max_epu32(dv14, dv16);
		dv9  = t1;
		dv10 = t2;
		dv13 = t3;
		dv14 = t4;
		
		// distance 16 swaps - just compare the vecs
		t1 =  _mm512_min_epu32(dv1, dv2);
		t2 =  _mm512_min_epu32(dv3, dv4);
		t3 =  _mm512_min_epu32(dv5, dv6);
		t4 =  _mm512_min_epu32(dv7, dv8);
		dv2 = _mm512_max_epu32(dv1, dv2);
		dv4 = _mm512_max_epu32(dv3, dv4);
		dv6 = _mm512_max_epu32(dv5, dv6);
		dv8 = _mm512_max_epu32(dv7, dv8);
		dv1 = t1;
		dv3 = t2;
		dv5 = t3;
		dv7 = t4;

		t1  =  _mm512_min_epu32(dv9 , dv10);
		t2  =  _mm512_min_epu32(dv11, dv12);
		t3  =  _mm512_min_epu32(dv13, dv14);
		t4  =  _mm512_min_epu32(dv15, dv16);
		dv10 = _mm512_max_epu32(dv9 , dv10);
		dv12 = _mm512_max_epu32(dv11, dv12);
		dv14 = _mm512_max_epu32(dv13, dv14);
		dv16 = _mm512_max_epu32(dv15, dv16);
		dv9  = t1;
		dv11 = t2;
		dv13 = t3;
		dv15 = t4;
		
		// distance 8 swaps - 256 bits
		dv1_swap = SWAP256(dv1);
		dv2_swap = SWAP256(dv2);
		dv3_swap = SWAP256(dv3);
		dv4_swap = SWAP256(dv4);
		dv5_swap = SWAP256(dv5);
		dv6_swap = SWAP256(dv6);
		dv7_swap = SWAP256(dv7);
		dv8_swap = SWAP256(dv8);

		// '00FF' for the swap between dist-8 lanes
		//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		//BLENDMASK(0xFF00, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		MINMAX(0xFF00, 0x00ff, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		
		dv1_swap = SWAP256(dv9);
		dv2_swap = SWAP256(dv10);
		dv3_swap = SWAP256(dv11);
		dv4_swap = SWAP256(dv12);
		dv5_swap = SWAP256(dv13);
		dv6_swap = SWAP256(dv14);
		dv7_swap = SWAP256(dv15);
		dv8_swap = SWAP256(dv16);
		
		//CMPGT(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
		//BLENDMASK(0xff00, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
		MINMAX(0xFF00, 0x00ff, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
		
		// distance 4 swaps
		dv1_swap = SWAP128(dv1);
		dv2_swap = SWAP128(dv2);
		dv3_swap = SWAP128(dv3);
		dv4_swap = SWAP128(dv4);
		dv5_swap = SWAP128(dv5);
		dv6_swap = SWAP128(dv6);
		dv7_swap = SWAP128(dv7);
		dv8_swap = SWAP128(dv8);

		// '0F' for the swap between dist-4 lanes, non alternating
		//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		//BLENDMASK(0xF0F0, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		MINMAX(0xf0f0, 0x0f0f, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		
		dv1_swap = SWAP128(dv9);
		dv2_swap = SWAP128(dv10);
		dv3_swap = SWAP128(dv11);
		dv4_swap = SWAP128(dv12);
		dv5_swap = SWAP128(dv13);
		dv6_swap = SWAP128(dv14);
		dv7_swap = SWAP128(dv15);
		dv8_swap = SWAP128(dv16);
		
		//CMPGT(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
		//BLENDMASK(0xf0f0, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
		MINMAX(0xf0f0, 0x0f0f, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);

		// distance 2 swaps
		dv1_swap = SWAP64(dv1);
		dv2_swap = SWAP64(dv2);
		dv3_swap = SWAP64(dv3);
		dv4_swap = SWAP64(dv4);
		dv5_swap = SWAP64(dv5);
		dv6_swap = SWAP64(dv6);
		dv7_swap = SWAP64(dv7);
		dv8_swap = SWAP64(dv8);

		// 'CC' for the swap between dist-2 lanes, non alternating
		//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		//BLENDMASK(0xCCCC, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		MINMAX(0xCCCC, 0x3333, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		
		dv1_swap = SWAP64(dv9);
		dv2_swap = SWAP64(dv10);
		dv3_swap = SWAP64(dv11);
		dv4_swap = SWAP64(dv12);
		dv5_swap = SWAP64(dv13);
		dv6_swap = SWAP64(dv14);
		dv7_swap = SWAP64(dv15);
		dv8_swap = SWAP64(dv16);
		
		//CMPGT(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
		//BLENDMASK(0xcccc, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
		MINMAX(0xCCCC, 0x3333, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
		
		// adjacent swaps
		dv1_swap = SWAP32(dv1);
		dv2_swap = SWAP32(dv2);
		dv3_swap = SWAP32(dv3);
		dv4_swap = SWAP32(dv4);
		dv5_swap = SWAP32(dv5);
		dv6_swap = SWAP32(dv6);
		dv7_swap = SWAP32(dv7);
		dv8_swap = SWAP32(dv8);

		// 'AA' for the swap between adjacent lanes
		//CMPGT(dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		//BLENDMASK(0xAAAA, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		MINMAX(0xaaaa, 0x5555, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		
		dv1_swap = SWAP32(dv9);
		dv2_swap = SWAP32(dv10);
		dv3_swap = SWAP32(dv11);
		dv4_swap = SWAP32(dv12);
		dv5_swap = SWAP32(dv13);
		dv6_swap = SWAP32(dv14);
		dv7_swap = SWAP32(dv15);
		dv8_swap = SWAP32(dv16);
		
		//CMPGT(dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
		//BLENDMASK(0xaaaa, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);
		MINMAX(0xaaaa, 0x5555, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16);

	}

	_mm512_store_epi32(data, dv1);
	_mm512_store_epi32(data + 16, dv2);
	_mm512_store_epi32(data + 32, dv3);
	_mm512_store_epi32(data + 48, dv4);
	_mm512_store_epi32(data + 64, dv5);
	_mm512_store_epi32(data + 80, dv6);
	_mm512_store_epi32(data + 96, dv7);
	_mm512_store_epi32(data + 112, dv8);
	_mm512_store_epi32(data + 128    , dv9);
	_mm512_store_epi32(data + 9 * 16 , dv10);
	_mm512_store_epi32(data + 10 * 16, dv11);
	_mm512_store_epi32(data + 11 * 16, dv12);
	_mm512_store_epi32(data + 12 * 16, dv13);
	_mm512_store_epi32(data + 13 * 16, dv14);
	_mm512_store_epi32(data + 14 * 16, dv15);
	_mm512_store_epi32(data + 15 * 16, dv16);
	
	return;
}

void bitonic_merge_dir_64(uint64_t* data, int dir)
{
	// perform the final merge pass on 64 elements
	__m512i t1;
	__m512i t2;
	__m512i t3;
	__m512i t4;
	__m512i t5;
	__m512i t6;
	__m512i t7;
	__m512i t8;
	__m512i dv1;
	__m512i dv2;
	__m512i dv3;
	__m512i dv4;
	__m512i dv5;
	__m512i dv6;
	__m512i dv7;
	__m512i dv8;
	__m512i dv1_swap;
	__m512i dv2_swap;
	__m512i dv3_swap;
	__m512i dv4_swap;
	__m512i dv5_swap;
	__m512i dv6_swap;
	__m512i dv7_swap;
	__m512i dv8_swap;
	__mmask8 m1;
	__mmask8 m2;
	__mmask8 m3;
	__mmask8 m4;
	__mmask8 m5;
	__mmask8 m6;
	__mmask8 m7;
	__mmask8 m8;

	dv1 = _mm512_load_epi64(data);
	dv2 = _mm512_load_epi64(data + 8);
	dv3 = _mm512_load_epi64(data + 16);
	dv4 = _mm512_load_epi64(data + 24);
	dv5 = _mm512_load_epi64(data + 32);
	dv6 = _mm512_load_epi64(data + 40);
	dv7 = _mm512_load_epi64(data + 48);
	dv8 = _mm512_load_epi64(data + 56);
	
	
	if (dir == 1)
	{
		// distance 32 swaps - just compare the vecs
		m1 = _mm512_cmplt_epu64_mask(dv1, dv5);
		m2 = _mm512_cmplt_epu64_mask(dv2, dv6);
		m3 = _mm512_cmplt_epu64_mask(dv3, dv7);
		m4 = _mm512_cmplt_epu64_mask(dv4, dv8);

		t1 = _mm512_mask_blend_epi64(m1, dv1, dv5);
		t2 = _mm512_mask_blend_epi64(m2, dv2, dv6);
		t3 = _mm512_mask_blend_epi64(m3, dv3, dv7);
		t4 = _mm512_mask_blend_epi64(m4, dv4, dv8);
		dv5 = _mm512_mask_blend_epi64(m1, dv5, dv1);
		dv6 = _mm512_mask_blend_epi64(m2, dv6, dv2);
		dv7 = _mm512_mask_blend_epi64(m3, dv7, dv3);
		dv8 = _mm512_mask_blend_epi64(m4, dv8, dv4);
		dv1 = t1;
		dv2 = t2;
		dv3 = t3;
		dv4 = t4;
		
		// distance 16 swaps - just compare the vecs
		m1 = _mm512_cmplt_epu64_mask(dv1, dv3);
		m2 = _mm512_cmplt_epu64_mask(dv2, dv4);
		m3 = _mm512_cmplt_epu64_mask(dv5, dv7);
		m4 = _mm512_cmplt_epu64_mask(dv6, dv8);

		t1 = _mm512_mask_blend_epi64(m1, dv1, dv3);
		t2 = _mm512_mask_blend_epi64(m2, dv2, dv4);
		t3 = _mm512_mask_blend_epi64(m3, dv5, dv7);
		t4 = _mm512_mask_blend_epi64(m4, dv6, dv8);

		dv3 = _mm512_mask_blend_epi64(m1, dv3, dv1);
		dv4 = _mm512_mask_blend_epi64(m2, dv4, dv2);
		dv7 = _mm512_mask_blend_epi64(m3, dv7, dv5);
		dv8 = _mm512_mask_blend_epi64(m4, dv8, dv6);

		dv1 = t1;
		dv2 = t2;
		dv5 = t3;
		dv6 = t4;

		// distance 8 swaps - just compare the vecs
		m1 = _mm512_cmplt_epu64_mask(dv1, dv2);
		m2 = _mm512_cmplt_epu64_mask(dv3, dv4);
		m3 = _mm512_cmplt_epu64_mask(dv5, dv6);
		m4 = _mm512_cmplt_epu64_mask(dv7, dv8);

		t1 = _mm512_mask_blend_epi64(m1, dv1, dv2);
		t2 = _mm512_mask_blend_epi64(m2, dv3, dv4);
		t3 = _mm512_mask_blend_epi64(m3, dv5, dv6);
		t4 = _mm512_mask_blend_epi64(m4, dv7, dv8);
		
		dv2 = _mm512_mask_blend_epi64(m1, dv2, dv1);
		dv4 = _mm512_mask_blend_epi64(m2, dv4, dv3);
		dv6 = _mm512_mask_blend_epi64(m3, dv6, dv5);
		dv8 = _mm512_mask_blend_epi64(m4, dv8, dv7);
		
		dv1 = t1;
		dv3 = t2;
		dv5 = t3;
		dv7 = t4;

		// distance 4 swaps
		dv1_swap = SWAP256(dv1);
		dv2_swap = SWAP256(dv2);
		dv3_swap = SWAP256(dv3);
		dv4_swap = SWAP256(dv4);
		dv5_swap = SWAP256(dv5);
		dv6_swap = SWAP256(dv6);
		dv7_swap = SWAP256(dv7);
		dv8_swap = SWAP256(dv8);
		
		m1 = _mm512_cmplt_epu64_mask(dv1, dv1_swap);
		m2 = _mm512_cmplt_epu64_mask(dv2, dv2_swap);
		m3 = _mm512_cmplt_epu64_mask(dv3, dv3_swap);
		m4 = _mm512_cmplt_epu64_mask(dv4, dv4_swap);
		m5 = _mm512_cmplt_epu64_mask(dv5, dv5_swap);
		m6 = _mm512_cmplt_epu64_mask(dv6, dv6_swap);
		m7 = _mm512_cmplt_epu64_mask(dv7, dv7_swap);
		m8 = _mm512_cmplt_epu64_mask(dv8, dv8_swap);

		// 'F0' for the swap between dist-4 lanes, non alternating
		dv1 = _mm512_mask_blend_epi64(m1 ^ 0xF0, dv1, dv1_swap);
		dv2 = _mm512_mask_blend_epi64(m2 ^ 0xF0, dv2, dv2_swap);
		dv3 = _mm512_mask_blend_epi64(m3 ^ 0xF0, dv3, dv3_swap);
		dv4 = _mm512_mask_blend_epi64(m4 ^ 0xF0, dv4, dv4_swap);
		dv5 = _mm512_mask_blend_epi64(m5 ^ 0xF0, dv5, dv5_swap);
		dv6 = _mm512_mask_blend_epi64(m6 ^ 0xF0, dv6, dv6_swap);
		dv7 = _mm512_mask_blend_epi64(m7 ^ 0xF0, dv7, dv7_swap);
		dv8 = _mm512_mask_blend_epi64(m8 ^ 0xF0, dv8, dv8_swap);

		// distance 2 swaps
		dv1_swap = SWAP128(dv1);
		dv2_swap = SWAP128(dv2);
		dv3_swap = SWAP128(dv3);
		dv4_swap = SWAP128(dv4);
		dv5_swap = SWAP128(dv5);
		dv6_swap = SWAP128(dv6);
		dv7_swap = SWAP128(dv7);
		dv8_swap = SWAP128(dv8);
		
		m1 = _mm512_cmplt_epu64_mask(dv1, dv1_swap);
		m2 = _mm512_cmplt_epu64_mask(dv2, dv2_swap);
		m3 = _mm512_cmplt_epu64_mask(dv3, dv3_swap);
		m4 = _mm512_cmplt_epu64_mask(dv4, dv4_swap);
		m5 = _mm512_cmplt_epu64_mask(dv5, dv5_swap);
		m6 = _mm512_cmplt_epu64_mask(dv6, dv6_swap);
		m7 = _mm512_cmplt_epu64_mask(dv7, dv7_swap);
		m8 = _mm512_cmplt_epu64_mask(dv8, dv8_swap);

		// 'CC' for the swap between dist-2 lanes, non alternating
		dv1 = _mm512_mask_blend_epi64(m1 ^ 0xCC, dv1, dv1_swap);
		dv2 = _mm512_mask_blend_epi64(m2 ^ 0xCC, dv2, dv2_swap);
		dv3 = _mm512_mask_blend_epi64(m3 ^ 0xCC, dv3, dv3_swap);
		dv4 = _mm512_mask_blend_epi64(m4 ^ 0xCC, dv4, dv4_swap);
		dv5 = _mm512_mask_blend_epi64(m5 ^ 0xCC, dv5, dv5_swap);
		dv6 = _mm512_mask_blend_epi64(m6 ^ 0xCC, dv6, dv6_swap);
		dv7 = _mm512_mask_blend_epi64(m7 ^ 0xCC, dv7, dv7_swap);
		dv8 = _mm512_mask_blend_epi64(m8 ^ 0xCC, dv8, dv8_swap);

		// adjacent swaps
		dv1_swap = SWAP64(dv1);
		dv2_swap = SWAP64(dv2);
		dv3_swap = SWAP64(dv3);
		dv4_swap = SWAP64(dv4);
		dv5_swap = SWAP64(dv5);
		dv6_swap = SWAP64(dv6);
		dv7_swap = SWAP64(dv7);
		dv8_swap = SWAP64(dv8);
		
		m1 = _mm512_cmplt_epu64_mask(dv1, dv1_swap);
		m2 = _mm512_cmplt_epu64_mask(dv2, dv2_swap);
		m3 = _mm512_cmplt_epu64_mask(dv3, dv3_swap);
		m4 = _mm512_cmplt_epu64_mask(dv4, dv4_swap);
		m5 = _mm512_cmplt_epu64_mask(dv5, dv5_swap);
		m6 = _mm512_cmplt_epu64_mask(dv6, dv6_swap);
		m7 = _mm512_cmplt_epu64_mask(dv7, dv7_swap);
		m8 = _mm512_cmplt_epu64_mask(dv8, dv8_swap);

		// 'AA' for the swap between adjacent lanes
		dv1 = _mm512_mask_blend_epi64(m1 ^ 0xAA, dv1, dv1_swap);
		dv2 = _mm512_mask_blend_epi64(m2 ^ 0xAA, dv2, dv2_swap);
		dv3 = _mm512_mask_blend_epi64(m3 ^ 0xAA, dv3, dv3_swap);
		dv4 = _mm512_mask_blend_epi64(m4 ^ 0xAA, dv4, dv4_swap);
		dv5 = _mm512_mask_blend_epi64(m5 ^ 0xAA, dv5, dv5_swap);
		dv6 = _mm512_mask_blend_epi64(m6 ^ 0xAA, dv6, dv6_swap);
		dv7 = _mm512_mask_blend_epi64(m7 ^ 0xAA, dv7, dv7_swap);
		dv8 = _mm512_mask_blend_epi64(m8 ^ 0xAA, dv8, dv8_swap);

	}
	else
	{

		// distance 32 swaps - just compare the vecs
		m1 = _mm512_cmpgt_epu64_mask(dv1, dv5);
		m2 = _mm512_cmpgt_epu64_mask(dv2, dv6);
		m3 = _mm512_cmpgt_epu64_mask(dv3, dv7);
		m4 = _mm512_cmpgt_epu64_mask(dv4, dv8);

		t1 = _mm512_mask_blend_epi64(m1, dv1, dv5);
		t2 = _mm512_mask_blend_epi64(m2, dv2, dv6);
		t3 = _mm512_mask_blend_epi64(m3, dv3, dv7);
		t4 = _mm512_mask_blend_epi64(m4, dv4, dv8);
		dv5 = _mm512_mask_blend_epi64(m1, dv5, dv1);
		dv6 = _mm512_mask_blend_epi64(m2, dv6, dv2);
		dv7 = _mm512_mask_blend_epi64(m3, dv7, dv3);
		dv8 = _mm512_mask_blend_epi64(m4, dv8, dv4);
		dv1 = t1;
		dv2 = t2;
		dv3 = t3;
		dv4 = t4;
		
		// distance 16 swaps - just compare the vecs
		m1 = _mm512_cmpgt_epu64_mask(dv1, dv3);
		m2 = _mm512_cmpgt_epu64_mask(dv2, dv4);
		m3 = _mm512_cmpgt_epu64_mask(dv5, dv7);
		m4 = _mm512_cmpgt_epu64_mask(dv6, dv8);

		t1 = _mm512_mask_blend_epi64(m1, dv1, dv3);
		t2 = _mm512_mask_blend_epi64(m2, dv2, dv4);
		t3 = _mm512_mask_blend_epi64(m3, dv5, dv7);
		t4 = _mm512_mask_blend_epi64(m4, dv6, dv8);

		dv3 = _mm512_mask_blend_epi64(m1, dv3, dv1);
		dv4 = _mm512_mask_blend_epi64(m2, dv4, dv2);
		dv7 = _mm512_mask_blend_epi64(m3, dv7, dv5);
		dv8 = _mm512_mask_blend_epi64(m4, dv8, dv6);

		dv1 = t1;
		dv2 = t2;
		dv5 = t3;
		dv6 = t4;

		// distance 8 swaps - just compare the vecs
		m1 = _mm512_cmpgt_epu64_mask(dv1, dv2);
		m2 = _mm512_cmpgt_epu64_mask(dv3, dv4);
		m3 = _mm512_cmpgt_epu64_mask(dv5, dv6);
		m4 = _mm512_cmpgt_epu64_mask(dv7, dv8);

		t1 = _mm512_mask_blend_epi64(m1, dv1, dv2);
		t2 = _mm512_mask_blend_epi64(m2, dv3, dv4);
		t3 = _mm512_mask_blend_epi64(m3, dv5, dv6);
		t4 = _mm512_mask_blend_epi64(m4, dv7, dv8);
		
		dv2 = _mm512_mask_blend_epi64(m1, dv2, dv1);
		dv4 = _mm512_mask_blend_epi64(m2, dv4, dv3);
		dv6 = _mm512_mask_blend_epi64(m3, dv6, dv5);
		dv8 = _mm512_mask_blend_epi64(m4, dv8, dv7);
		
		dv1 = t1;
		dv3 = t2;
		dv5 = t3;
		dv7 = t4;

		// distance 4 swaps
		dv1_swap = SWAP256(dv1);
		dv2_swap = SWAP256(dv2);
		dv3_swap = SWAP256(dv3);
		dv4_swap = SWAP256(dv4);
		dv5_swap = SWAP256(dv5);
		dv6_swap = SWAP256(dv6);
		dv7_swap = SWAP256(dv7);
		dv8_swap = SWAP256(dv8);
		
		m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
		m2 = _mm512_cmpgt_epu64_mask(dv2, dv2_swap);
		m3 = _mm512_cmpgt_epu64_mask(dv3, dv3_swap);
		m4 = _mm512_cmpgt_epu64_mask(dv4, dv4_swap);
		m5 = _mm512_cmpgt_epu64_mask(dv5, dv5_swap);
		m6 = _mm512_cmpgt_epu64_mask(dv6, dv6_swap);
		m7 = _mm512_cmpgt_epu64_mask(dv7, dv7_swap);
		m8 = _mm512_cmpgt_epu64_mask(dv8, dv8_swap);

		// 'F0' for the swap between dist-4 lanes, non alternating
		dv1 = _mm512_mask_blend_epi64(m1 ^ 0xF0, dv1, dv1_swap);
		dv2 = _mm512_mask_blend_epi64(m2 ^ 0xF0, dv2, dv2_swap);
		dv3 = _mm512_mask_blend_epi64(m3 ^ 0xF0, dv3, dv3_swap);
		dv4 = _mm512_mask_blend_epi64(m4 ^ 0xF0, dv4, dv4_swap);
		dv5 = _mm512_mask_blend_epi64(m5 ^ 0xF0, dv5, dv5_swap);
		dv6 = _mm512_mask_blend_epi64(m6 ^ 0xF0, dv6, dv6_swap);
		dv7 = _mm512_mask_blend_epi64(m7 ^ 0xF0, dv7, dv7_swap);
		dv8 = _mm512_mask_blend_epi64(m8 ^ 0xF0, dv8, dv8_swap);

		// distance 2 swaps
		dv1_swap = SWAP128(dv1);
		dv2_swap = SWAP128(dv2);
		dv3_swap = SWAP128(dv3);
		dv4_swap = SWAP128(dv4);
		dv5_swap = SWAP128(dv5);
		dv6_swap = SWAP128(dv6);
		dv7_swap = SWAP128(dv7);
		dv8_swap = SWAP128(dv8);
		
		m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
		m2 = _mm512_cmpgt_epu64_mask(dv2, dv2_swap);
		m3 = _mm512_cmpgt_epu64_mask(dv3, dv3_swap);
		m4 = _mm512_cmpgt_epu64_mask(dv4, dv4_swap);
		m5 = _mm512_cmpgt_epu64_mask(dv5, dv5_swap);
		m6 = _mm512_cmpgt_epu64_mask(dv6, dv6_swap);
		m7 = _mm512_cmpgt_epu64_mask(dv7, dv7_swap);
		m8 = _mm512_cmpgt_epu64_mask(dv8, dv8_swap);

		// 'CC' for the swap between dist-2 lanes, non alternating
		dv1 = _mm512_mask_blend_epi64(m1 ^ 0xCC, dv1, dv1_swap);
		dv2 = _mm512_mask_blend_epi64(m2 ^ 0xCC, dv2, dv2_swap);
		dv3 = _mm512_mask_blend_epi64(m3 ^ 0xCC, dv3, dv3_swap);
		dv4 = _mm512_mask_blend_epi64(m4 ^ 0xCC, dv4, dv4_swap);
		dv5 = _mm512_mask_blend_epi64(m5 ^ 0xCC, dv5, dv5_swap);
		dv6 = _mm512_mask_blend_epi64(m6 ^ 0xCC, dv6, dv6_swap);
		dv7 = _mm512_mask_blend_epi64(m7 ^ 0xCC, dv7, dv7_swap);
		dv8 = _mm512_mask_blend_epi64(m8 ^ 0xCC, dv8, dv8_swap);

		// adjacent swaps
		dv1_swap = SWAP64(dv1);
		dv2_swap = SWAP64(dv2);
		dv3_swap = SWAP64(dv3);
		dv4_swap = SWAP64(dv4);
		dv5_swap = SWAP64(dv5);
		dv6_swap = SWAP64(dv6);
		dv7_swap = SWAP64(dv7);
		dv8_swap = SWAP64(dv8);
		
		m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
		m2 = _mm512_cmpgt_epu64_mask(dv2, dv2_swap);
		m3 = _mm512_cmpgt_epu64_mask(dv3, dv3_swap);
		m4 = _mm512_cmpgt_epu64_mask(dv4, dv4_swap);
		m5 = _mm512_cmpgt_epu64_mask(dv5, dv5_swap);
		m6 = _mm512_cmpgt_epu64_mask(dv6, dv6_swap);
		m7 = _mm512_cmpgt_epu64_mask(dv7, dv7_swap);
		m8 = _mm512_cmpgt_epu64_mask(dv8, dv8_swap);

		// 'AA' for the swap between adjacent lanes
		dv1 = _mm512_mask_blend_epi64(m1 ^ 0xAA, dv1, dv1_swap);
		dv2 = _mm512_mask_blend_epi64(m2 ^ 0xAA, dv2, dv2_swap);
		dv3 = _mm512_mask_blend_epi64(m3 ^ 0xAA, dv3, dv3_swap);
		dv4 = _mm512_mask_blend_epi64(m4 ^ 0xAA, dv4, dv4_swap);
		dv5 = _mm512_mask_blend_epi64(m5 ^ 0xAA, dv5, dv5_swap);
		dv6 = _mm512_mask_blend_epi64(m6 ^ 0xAA, dv6, dv6_swap);
		dv7 = _mm512_mask_blend_epi64(m7 ^ 0xAA, dv7, dv7_swap);
		dv8 = _mm512_mask_blend_epi64(m8 ^ 0xAA, dv8, dv8_swap);
	}

	_mm512_store_epi64(data, dv1);
	_mm512_store_epi64(data + 8, dv2);
	_mm512_store_epi64(data + 16, dv3);
	_mm512_store_epi64(data + 24, dv4);
	_mm512_store_epi64(data + 32, dv5);
	_mm512_store_epi64(data + 40, dv6);
	_mm512_store_epi64(data + 48, dv7);
	_mm512_store_epi64(data + 56, dv8);
	
	return;
}

void bitonic_sort_dir_64(uint64_t* data, int dir) 
{
	int i, j;

	__m512i t1;
	__m512i t2;
	__m512i t3;
	__m512i t4;
	__m512i t5;
	__m512i t6;
	__m512i t7;
	__m512i t8;
	__m512i dv1;
	__m512i dv2;
	__m512i dv3;
	__m512i dv4;
	__m512i dv5;
	__m512i dv6;
	__m512i dv7;
	__m512i dv8;
	__m512i dv1_swap;
	__m512i dv2_swap;
	__m512i dv3_swap;
	__m512i dv4_swap;
	__m512i dv5_swap;
	__m512i dv6_swap;
	__m512i dv7_swap;
	__m512i dv8_swap;
	__mmask8 m1;
	__mmask8 m2;
	__mmask8 m3;
	__mmask8 m4;
	__mmask8 m5;
	__mmask8 m6;
	__mmask8 m7;
	__mmask8 m8;

	dv1 = _mm512_load_epi64(data);
	dv2 = _mm512_load_epi64(data + 8);
	dv3 = _mm512_load_epi64(data + 16);
	dv4 = _mm512_load_epi64(data + 24);
	dv5 = _mm512_load_epi64(data + 32);
	dv6 = _mm512_load_epi64(data + 40);
	dv7 = _mm512_load_epi64(data + 48);
	dv8 = _mm512_load_epi64(data + 56);

	// phase 1 : dist-2 alternating compares
	
	// adjacent swaps, alternating compares
	dv1_swap = SWAP64(dv1);
	dv2_swap = SWAP64(dv2);
	dv3_swap = SWAP64(dv3);
	dv4_swap = SWAP64(dv4);
	dv5_swap = SWAP64(dv5);
	dv6_swap = SWAP64(dv6);
	dv7_swap = SWAP64(dv7);
	dv8_swap = SWAP64(dv8);
	
	m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
	m2 = _mm512_cmpgt_epu64_mask(dv2, dv2_swap);
	m3 = _mm512_cmpgt_epu64_mask(dv3, dv3_swap);
	m4 = _mm512_cmpgt_epu64_mask(dv4, dv4_swap);
	m5 = _mm512_cmpgt_epu64_mask(dv5, dv5_swap);
	m6 = _mm512_cmpgt_epu64_mask(dv6, dv6_swap);
	m7 = _mm512_cmpgt_epu64_mask(dv7, dv7_swap);
	m8 = _mm512_cmpgt_epu64_mask(dv8, dv8_swap);
	
	// 'AA' for the swap between adjacent lanes ^ 'CC' to alternate gt vs. lt --> 0x66
	dv1 = _mm512_mask_blend_epi64(m1 ^ 0x66, dv1, dv1_swap);
	dv2 = _mm512_mask_blend_epi64(m2 ^ 0x66, dv2, dv2_swap);
	dv3 = _mm512_mask_blend_epi64(m3 ^ 0x66, dv3, dv3_swap);
	dv4 = _mm512_mask_blend_epi64(m4 ^ 0x66, dv4, dv4_swap);
	dv5 = _mm512_mask_blend_epi64(m5 ^ 0x66, dv5, dv5_swap);
	dv6 = _mm512_mask_blend_epi64(m6 ^ 0x66, dv6, dv6_swap);
	dv7 = _mm512_mask_blend_epi64(m7 ^ 0x66, dv7, dv7_swap);
	dv8 = _mm512_mask_blend_epi64(m8 ^ 0x66, dv8, dv8_swap);
	
	// phase 2 : dist-4 alternating compares

	// distance 2 swaps
	dv1_swap = SWAP128(dv1);
	dv2_swap = SWAP128(dv2);
	dv3_swap = SWAP128(dv3);
	dv4_swap = SWAP128(dv4);
	dv5_swap = SWAP128(dv5);
	dv6_swap = SWAP128(dv6);
	dv7_swap = SWAP128(dv7);
	dv8_swap = SWAP128(dv8);
	
	m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
	m2 = _mm512_cmpgt_epu64_mask(dv2, dv2_swap);
	m3 = _mm512_cmpgt_epu64_mask(dv3, dv3_swap);
	m4 = _mm512_cmpgt_epu64_mask(dv4, dv4_swap);
	m5 = _mm512_cmpgt_epu64_mask(dv5, dv5_swap);
	m6 = _mm512_cmpgt_epu64_mask(dv6, dv6_swap);
	m7 = _mm512_cmpgt_epu64_mask(dv7, dv7_swap);
	m8 = _mm512_cmpgt_epu64_mask(dv8, dv8_swap);

	// 'CC' for the swap between dist-2 lanes ^ 'F0' to alternate gt vs. lt --> 0xC3
	dv1 = _mm512_mask_blend_epi64(m1 ^ 0x3C, dv1, dv1_swap);
	dv2 = _mm512_mask_blend_epi64(m2 ^ 0x3C, dv2, dv2_swap);
	dv3 = _mm512_mask_blend_epi64(m3 ^ 0x3C, dv3, dv3_swap);
	dv4 = _mm512_mask_blend_epi64(m4 ^ 0x3C, dv4, dv4_swap);
	dv5 = _mm512_mask_blend_epi64(m5 ^ 0x3C, dv5, dv5_swap);
	dv6 = _mm512_mask_blend_epi64(m6 ^ 0x3C, dv6, dv6_swap);
	dv7 = _mm512_mask_blend_epi64(m7 ^ 0x3C, dv7, dv7_swap);
	dv8 = _mm512_mask_blend_epi64(m8 ^ 0x3C, dv8, dv8_swap);

	// adjacent swaps
	dv1_swap = SWAP64(dv1);
	dv2_swap = SWAP64(dv2);
	dv3_swap = SWAP64(dv3);
	dv4_swap = SWAP64(dv4);
	dv5_swap = SWAP64(dv5);
	dv6_swap = SWAP64(dv6);
	dv7_swap = SWAP64(dv7);
	dv8_swap = SWAP64(dv8);
	
	m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
	m2 = _mm512_cmpgt_epu64_mask(dv2, dv2_swap);
	m3 = _mm512_cmpgt_epu64_mask(dv3, dv3_swap);
	m4 = _mm512_cmpgt_epu64_mask(dv4, dv4_swap);
	m5 = _mm512_cmpgt_epu64_mask(dv5, dv5_swap);
	m6 = _mm512_cmpgt_epu64_mask(dv6, dv6_swap);
	m7 = _mm512_cmpgt_epu64_mask(dv7, dv7_swap);
	m8 = _mm512_cmpgt_epu64_mask(dv8, dv8_swap);

	// 'AA' for the swap between adjacent lanes ^ 'F0' to alternate gt vs. lt --> 0xA5
	dv1 = _mm512_mask_blend_epi64(m1 ^ 0x5A, dv1, dv1_swap);
	dv2 = _mm512_mask_blend_epi64(m2 ^ 0x5A, dv2, dv2_swap);
	dv3 = _mm512_mask_blend_epi64(m3 ^ 0x5A, dv3, dv3_swap);
	dv4 = _mm512_mask_blend_epi64(m4 ^ 0x5A, dv4, dv4_swap);
	dv5 = _mm512_mask_blend_epi64(m5 ^ 0x5A, dv5, dv5_swap);
	dv6 = _mm512_mask_blend_epi64(m6 ^ 0x5A, dv6, dv6_swap);
	dv7 = _mm512_mask_blend_epi64(m7 ^ 0x5A, dv7, dv7_swap);
	dv8 = _mm512_mask_blend_epi64(m8 ^ 0x5A, dv8, dv8_swap);


	// phase 3 : dist-8 alternating compares

	// distance 4 swaps
	dv1_swap = SWAP256(dv1);
	dv2_swap = SWAP256(dv2);
	dv3_swap = SWAP256(dv3);
	dv4_swap = SWAP256(dv4);
	dv5_swap = SWAP256(dv5);
	dv6_swap = SWAP256(dv6);
	dv7_swap = SWAP256(dv7);
	dv8_swap = SWAP256(dv8);
	
	m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
	m2 = _mm512_cmplt_epu64_mask(dv2, dv2_swap);
	m3 = _mm512_cmpgt_epu64_mask(dv3, dv3_swap);
	m4 = _mm512_cmplt_epu64_mask(dv4, dv4_swap);
	m5 = _mm512_cmpgt_epu64_mask(dv5, dv5_swap);
	m6 = _mm512_cmplt_epu64_mask(dv6, dv6_swap);
	m7 = _mm512_cmpgt_epu64_mask(dv7, dv7_swap);
	m8 = _mm512_cmplt_epu64_mask(dv8, dv8_swap);

	// '0F' for the swap between dist-4 lanes, non alternating
	dv1 = _mm512_mask_blend_epi64(m1 ^ 0xF0, dv1, dv1_swap);
	dv2 = _mm512_mask_blend_epi64(m2 ^ 0xF0, dv2, dv2_swap);
	dv3 = _mm512_mask_blend_epi64(m3 ^ 0xF0, dv3, dv3_swap);
	dv4 = _mm512_mask_blend_epi64(m4 ^ 0xF0, dv4, dv4_swap);
	dv5 = _mm512_mask_blend_epi64(m5 ^ 0xF0, dv5, dv5_swap);
	dv6 = _mm512_mask_blend_epi64(m6 ^ 0xF0, dv6, dv6_swap);
	dv7 = _mm512_mask_blend_epi64(m7 ^ 0xF0, dv7, dv7_swap);
	dv8 = _mm512_mask_blend_epi64(m8 ^ 0xF0, dv8, dv8_swap);

	// distance 2 swaps
	dv1_swap = SWAP128(dv1);
	dv2_swap = SWAP128(dv2);
	dv3_swap = SWAP128(dv3);
	dv4_swap = SWAP128(dv4);
	dv5_swap = SWAP128(dv5);
	dv6_swap = SWAP128(dv6);
	dv7_swap = SWAP128(dv7);
	dv8_swap = SWAP128(dv8);
	
	m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
	m2 = _mm512_cmplt_epu64_mask(dv2, dv2_swap);
	m3 = _mm512_cmpgt_epu64_mask(dv3, dv3_swap);
	m4 = _mm512_cmplt_epu64_mask(dv4, dv4_swap);
	m5 = _mm512_cmpgt_epu64_mask(dv5, dv5_swap);
	m6 = _mm512_cmplt_epu64_mask(dv6, dv6_swap);
	m7 = _mm512_cmpgt_epu64_mask(dv7, dv7_swap);
	m8 = _mm512_cmplt_epu64_mask(dv8, dv8_swap);

	// 'CC' for the swap between dist-2 lanes, non alternating
	dv1 = _mm512_mask_blend_epi64(m1 ^ 0xCC, dv1, dv1_swap);
	dv2 = _mm512_mask_blend_epi64(m2 ^ 0xCC, dv2, dv2_swap);
	dv3 = _mm512_mask_blend_epi64(m3 ^ 0xCC, dv3, dv3_swap);
	dv4 = _mm512_mask_blend_epi64(m4 ^ 0xCC, dv4, dv4_swap);
	dv5 = _mm512_mask_blend_epi64(m5 ^ 0xCC, dv5, dv5_swap);
	dv6 = _mm512_mask_blend_epi64(m6 ^ 0xCC, dv6, dv6_swap);
	dv7 = _mm512_mask_blend_epi64(m7 ^ 0xCC, dv7, dv7_swap);
	dv8 = _mm512_mask_blend_epi64(m8 ^ 0xCC, dv8, dv8_swap);

	// adjacent swaps
	dv1_swap = SWAP64(dv1);
	dv2_swap = SWAP64(dv2);
	dv3_swap = SWAP64(dv3);
	dv4_swap = SWAP64(dv4);
	dv5_swap = SWAP64(dv5);
	dv6_swap = SWAP64(dv6);
	dv7_swap = SWAP64(dv7);
	dv8_swap = SWAP64(dv8);
	
	m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
	m2 = _mm512_cmplt_epu64_mask(dv2, dv2_swap);
	m3 = _mm512_cmpgt_epu64_mask(dv3, dv3_swap);
	m4 = _mm512_cmplt_epu64_mask(dv4, dv4_swap);
	m5 = _mm512_cmpgt_epu64_mask(dv5, dv5_swap);
	m6 = _mm512_cmplt_epu64_mask(dv6, dv6_swap);
	m7 = _mm512_cmpgt_epu64_mask(dv7, dv7_swap);
	m8 = _mm512_cmplt_epu64_mask(dv8, dv8_swap);

	// 'AA' for the swap between adjacent lanes, non alternating
	dv1 = _mm512_mask_blend_epi64(m1 ^ 0xAA, dv1, dv1_swap);
	dv2 = _mm512_mask_blend_epi64(m2 ^ 0xAA, dv2, dv2_swap);
	dv3 = _mm512_mask_blend_epi64(m3 ^ 0xAA, dv3, dv3_swap);
	dv4 = _mm512_mask_blend_epi64(m4 ^ 0xAA, dv4, dv4_swap);
	dv5 = _mm512_mask_blend_epi64(m5 ^ 0xAA, dv5, dv5_swap);
	dv6 = _mm512_mask_blend_epi64(m6 ^ 0xAA, dv6, dv6_swap);
	dv7 = _mm512_mask_blend_epi64(m7 ^ 0xAA, dv7, dv7_swap);
	dv8 = _mm512_mask_blend_epi64(m8 ^ 0xAA, dv8, dv8_swap);


	// phase 4 : dist-16 alternating compares

	// distance 8 swaps - just compare the vecs
	m1 = _mm512_cmpgt_epu64_mask(dv1, dv2);
	m3 = _mm512_cmplt_epu64_mask(dv3, dv4);
	m5 = _mm512_cmpgt_epu64_mask(dv5, dv6);
	m7 = _mm512_cmplt_epu64_mask(dv7, dv8);
	t1 = _mm512_mask_blend_epi64(m1, dv1, dv2);
	t2 = _mm512_mask_blend_epi64(m3, dv3, dv4);
	t3 = _mm512_mask_blend_epi64(m5, dv5, dv6);
	t4 = _mm512_mask_blend_epi64(m7, dv7, dv8);
	dv2 = _mm512_mask_blend_epi64(m1, dv2, dv1);
	dv4 = _mm512_mask_blend_epi64(m3, dv4, dv3);
	dv6 = _mm512_mask_blend_epi64(m5, dv6, dv5);
	dv8 = _mm512_mask_blend_epi64(m7, dv8, dv7);
	dv1 = t1;
	dv3 = t2;
	dv5 = t3;
	dv7 = t4;

	// distance 4 swaps
	dv1_swap = SWAP256(dv1);
	dv2_swap = SWAP256(dv2);
	dv3_swap = SWAP256(dv3);
	dv4_swap = SWAP256(dv4);
	dv5_swap = SWAP256(dv5);
	dv6_swap = SWAP256(dv6);
	dv7_swap = SWAP256(dv7);
	dv8_swap = SWAP256(dv8);
	
	m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
	m2 = _mm512_cmpgt_epu64_mask(dv2, dv2_swap);
	m3 = _mm512_cmplt_epu64_mask(dv3, dv3_swap);
	m4 = _mm512_cmplt_epu64_mask(dv4, dv4_swap);
	m5 = _mm512_cmpgt_epu64_mask(dv5, dv5_swap);
	m6 = _mm512_cmpgt_epu64_mask(dv6, dv6_swap);
	m7 = _mm512_cmplt_epu64_mask(dv7, dv7_swap);
	m8 = _mm512_cmplt_epu64_mask(dv8, dv8_swap);

	// '0F' for the swap between dist-4 lanes, non alternating
	dv1 = _mm512_mask_blend_epi64(m1 ^ 0xF0, dv1, dv1_swap);
	dv2 = _mm512_mask_blend_epi64(m2 ^ 0xF0, dv2, dv2_swap);
	dv3 = _mm512_mask_blend_epi64(m3 ^ 0xF0, dv3, dv3_swap);
	dv4 = _mm512_mask_blend_epi64(m4 ^ 0xF0, dv4, dv4_swap);
	dv5 = _mm512_mask_blend_epi64(m5 ^ 0xF0, dv5, dv5_swap);
	dv6 = _mm512_mask_blend_epi64(m6 ^ 0xF0, dv6, dv6_swap);
	dv7 = _mm512_mask_blend_epi64(m7 ^ 0xF0, dv7, dv7_swap);
	dv8 = _mm512_mask_blend_epi64(m8 ^ 0xF0, dv8, dv8_swap);

	// distance 2 swaps
	dv1_swap = SWAP128(dv1);
	dv2_swap = SWAP128(dv2);
	dv3_swap = SWAP128(dv3);
	dv4_swap = SWAP128(dv4);
	dv5_swap = SWAP128(dv5);
	dv6_swap = SWAP128(dv6);
	dv7_swap = SWAP128(dv7);
	dv8_swap = SWAP128(dv8);
	
	m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
	m2 = _mm512_cmpgt_epu64_mask(dv2, dv2_swap);
	m3 = _mm512_cmplt_epu64_mask(dv3, dv3_swap);
	m4 = _mm512_cmplt_epu64_mask(dv4, dv4_swap);
	m5 = _mm512_cmpgt_epu64_mask(dv5, dv5_swap);
	m6 = _mm512_cmpgt_epu64_mask(dv6, dv6_swap);
	m7 = _mm512_cmplt_epu64_mask(dv7, dv7_swap);
	m8 = _mm512_cmplt_epu64_mask(dv8, dv8_swap);

	// 'CC' for the swap between dist-2 lanes, non alternating
	dv1 = _mm512_mask_blend_epi64(m1 ^ 0xCC, dv1, dv1_swap);
	dv2 = _mm512_mask_blend_epi64(m2 ^ 0xCC, dv2, dv2_swap);
	dv3 = _mm512_mask_blend_epi64(m3 ^ 0xCC, dv3, dv3_swap);
	dv4 = _mm512_mask_blend_epi64(m4 ^ 0xCC, dv4, dv4_swap);
	dv5 = _mm512_mask_blend_epi64(m5 ^ 0xCC, dv5, dv5_swap);
	dv6 = _mm512_mask_blend_epi64(m6 ^ 0xCC, dv6, dv6_swap);
	dv7 = _mm512_mask_blend_epi64(m7 ^ 0xCC, dv7, dv7_swap);
	dv8 = _mm512_mask_blend_epi64(m8 ^ 0xCC, dv8, dv8_swap);

	// adjacent swaps
	dv1_swap = SWAP64(dv1);
	dv2_swap = SWAP64(dv2);
	dv3_swap = SWAP64(dv3);
	dv4_swap = SWAP64(dv4);
	dv5_swap = SWAP64(dv5);
	dv6_swap = SWAP64(dv6);
	dv7_swap = SWAP64(dv7);
	dv8_swap = SWAP64(dv8);
	
	m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
	m2 = _mm512_cmpgt_epu64_mask(dv2, dv2_swap);
	m3 = _mm512_cmplt_epu64_mask(dv3, dv3_swap);
	m4 = _mm512_cmplt_epu64_mask(dv4, dv4_swap);
	m5 = _mm512_cmpgt_epu64_mask(dv5, dv5_swap);
	m6 = _mm512_cmpgt_epu64_mask(dv6, dv6_swap);
	m7 = _mm512_cmplt_epu64_mask(dv7, dv7_swap);
	m8 = _mm512_cmplt_epu64_mask(dv8, dv8_swap);

	// 'AA' for the swap between adjacent lanes
	dv1 = _mm512_mask_blend_epi64(m1 ^ 0xAA, dv1, dv1_swap);
	dv2 = _mm512_mask_blend_epi64(m2 ^ 0xAA, dv2, dv2_swap);
	dv3 = _mm512_mask_blend_epi64(m3 ^ 0xAA, dv3, dv3_swap);
	dv4 = _mm512_mask_blend_epi64(m4 ^ 0xAA, dv4, dv4_swap);
	dv5 = _mm512_mask_blend_epi64(m5 ^ 0xAA, dv5, dv5_swap);
	dv6 = _mm512_mask_blend_epi64(m6 ^ 0xAA, dv6, dv6_swap);
	dv7 = _mm512_mask_blend_epi64(m7 ^ 0xAA, dv7, dv7_swap);
	dv8 = _mm512_mask_blend_epi64(m8 ^ 0xAA, dv8, dv8_swap);


	// phase 5 : dist-32 alternating compares

	// distance 16 swaps - just compare the vecs
	m1 = _mm512_cmpgt_epu64_mask(dv1, dv3);
	m2 = _mm512_cmpgt_epu64_mask(dv2, dv4);
	m3 = _mm512_cmplt_epu64_mask(dv5, dv7);
	m4 = _mm512_cmplt_epu64_mask(dv6, dv8);

	t1 = _mm512_mask_blend_epi64(m1, dv1, dv3);
	t2 = _mm512_mask_blend_epi64(m2, dv2, dv4);
	t3 = _mm512_mask_blend_epi64(m3, dv5, dv7);
	t4 = _mm512_mask_blend_epi64(m4, dv6, dv8);
		
	dv3 = _mm512_mask_blend_epi64(m1, dv3, dv1);
	dv4 = _mm512_mask_blend_epi64(m2, dv4, dv2);
	dv7 = _mm512_mask_blend_epi64(m3, dv7, dv5);
	dv8 = _mm512_mask_blend_epi64(m4, dv8, dv6);
	dv1 = t1;
	dv2 = t2;
	dv5 = t3;
	dv6 = t4;

	// distance 8 swaps - just compare the vecs
	m1 = _mm512_cmpgt_epu64_mask(dv1, dv2);
	m2 = _mm512_cmpgt_epu64_mask(dv3, dv4);
	m3 = _mm512_cmplt_epu64_mask(dv5, dv6);
	m4 = _mm512_cmplt_epu64_mask(dv7, dv8);

	t1 = _mm512_mask_blend_epi64(m1, dv1, dv2);
	t2 = _mm512_mask_blend_epi64(m2, dv3, dv4);
	t3 = _mm512_mask_blend_epi64(m3, dv5, dv6);
	t4 = _mm512_mask_blend_epi64(m4, dv7, dv8);
	
	dv2 = _mm512_mask_blend_epi64(m1, dv2, dv1);
	dv4 = _mm512_mask_blend_epi64(m2, dv4, dv3);
	dv6 = _mm512_mask_blend_epi64(m3, dv6, dv5);
	dv8 = _mm512_mask_blend_epi64(m4, dv8, dv7);
	
	dv1 = t1;
	dv3 = t2;
	dv5 = t3;
	dv7 = t4;

	// distance 4 swaps
	dv1_swap = SWAP256(dv1);
	dv2_swap = SWAP256(dv2);
	dv3_swap = SWAP256(dv3);
	dv4_swap = SWAP256(dv4);
	dv5_swap = SWAP256(dv5);
	dv6_swap = SWAP256(dv6);
	dv7_swap = SWAP256(dv7);
	dv8_swap = SWAP256(dv8);
	
	m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
	m2 = _mm512_cmpgt_epu64_mask(dv2, dv2_swap);
	m3 = _mm512_cmpgt_epu64_mask(dv3, dv3_swap);
	m4 = _mm512_cmpgt_epu64_mask(dv4, dv4_swap);
	m5 = _mm512_cmplt_epu64_mask(dv5, dv5_swap);
	m6 = _mm512_cmplt_epu64_mask(dv6, dv6_swap);
	m7 = _mm512_cmplt_epu64_mask(dv7, dv7_swap);
	m8 = _mm512_cmplt_epu64_mask(dv8, dv8_swap);

	// '0F' for the swap between dist-4 lanes, non alternating
	dv1 = _mm512_mask_blend_epi64(m1 ^ 0xF0, dv1, dv1_swap);
	dv2 = _mm512_mask_blend_epi64(m2 ^ 0xF0, dv2, dv2_swap);
	dv3 = _mm512_mask_blend_epi64(m3 ^ 0xF0, dv3, dv3_swap);
	dv4 = _mm512_mask_blend_epi64(m4 ^ 0xF0, dv4, dv4_swap);
	dv5 = _mm512_mask_blend_epi64(m5 ^ 0xF0, dv5, dv5_swap);
	dv6 = _mm512_mask_blend_epi64(m6 ^ 0xF0, dv6, dv6_swap);
	dv7 = _mm512_mask_blend_epi64(m7 ^ 0xF0, dv7, dv7_swap);
	dv8 = _mm512_mask_blend_epi64(m8 ^ 0xF0, dv8, dv8_swap);

	// distance 2 swaps
	dv1_swap = SWAP128(dv1);
	dv2_swap = SWAP128(dv2);
	dv3_swap = SWAP128(dv3);
	dv4_swap = SWAP128(dv4);
	dv5_swap = SWAP128(dv5);
	dv6_swap = SWAP128(dv6);
	dv7_swap = SWAP128(dv7);
	dv8_swap = SWAP128(dv8);
	
	m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
	m2 = _mm512_cmpgt_epu64_mask(dv2, dv2_swap);
	m3 = _mm512_cmpgt_epu64_mask(dv3, dv3_swap);
	m4 = _mm512_cmpgt_epu64_mask(dv4, dv4_swap);
	m5 = _mm512_cmplt_epu64_mask(dv5, dv5_swap);
	m6 = _mm512_cmplt_epu64_mask(dv6, dv6_swap);
	m7 = _mm512_cmplt_epu64_mask(dv7, dv7_swap);
	m8 = _mm512_cmplt_epu64_mask(dv8, dv8_swap);

	// 'CC' for the swap between dist-2 lanes, non alternating
	dv1 = _mm512_mask_blend_epi64(m1 ^ 0xCC, dv1, dv1_swap);
	dv2 = _mm512_mask_blend_epi64(m2 ^ 0xCC, dv2, dv2_swap);
	dv3 = _mm512_mask_blend_epi64(m3 ^ 0xCC, dv3, dv3_swap);
	dv4 = _mm512_mask_blend_epi64(m4 ^ 0xCC, dv4, dv4_swap);
	dv5 = _mm512_mask_blend_epi64(m5 ^ 0xCC, dv5, dv5_swap);
	dv6 = _mm512_mask_blend_epi64(m6 ^ 0xCC, dv6, dv6_swap);
	dv7 = _mm512_mask_blend_epi64(m7 ^ 0xCC, dv7, dv7_swap);
	dv8 = _mm512_mask_blend_epi64(m8 ^ 0xCC, dv8, dv8_swap);

	// adjacent swaps
	dv1_swap = SWAP64(dv1);
	dv2_swap = SWAP64(dv2);
	dv3_swap = SWAP64(dv3);
	dv4_swap = SWAP64(dv4);
	dv5_swap = SWAP64(dv5);
	dv6_swap = SWAP64(dv6);
	dv7_swap = SWAP64(dv7);
	dv8_swap = SWAP64(dv8);
	
	m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
	m2 = _mm512_cmpgt_epu64_mask(dv2, dv2_swap);
	m3 = _mm512_cmpgt_epu64_mask(dv3, dv3_swap);
	m4 = _mm512_cmpgt_epu64_mask(dv4, dv4_swap);
	m5 = _mm512_cmplt_epu64_mask(dv5, dv5_swap);
	m6 = _mm512_cmplt_epu64_mask(dv6, dv6_swap);
	m7 = _mm512_cmplt_epu64_mask(dv7, dv7_swap);
	m8 = _mm512_cmplt_epu64_mask(dv8, dv8_swap);

	// 'AA' for the swap between adjacent lanes
	dv1 = _mm512_mask_blend_epi64(m1 ^ 0xAA, dv1, dv1_swap);
	dv2 = _mm512_mask_blend_epi64(m2 ^ 0xAA, dv2, dv2_swap);
	dv3 = _mm512_mask_blend_epi64(m3 ^ 0xAA, dv3, dv3_swap);
	dv4 = _mm512_mask_blend_epi64(m4 ^ 0xAA, dv4, dv4_swap);
	dv5 = _mm512_mask_blend_epi64(m5 ^ 0xAA, dv5, dv5_swap);
	dv6 = _mm512_mask_blend_epi64(m6 ^ 0xAA, dv6, dv6_swap);
	dv7 = _mm512_mask_blend_epi64(m7 ^ 0xAA, dv7, dv7_swap);
	dv8 = _mm512_mask_blend_epi64(m8 ^ 0xAA, dv8, dv8_swap);


	if (dir == 1)
	{
		// distance 32 swaps - just compare the vecs
		m1 = _mm512_cmplt_epu64_mask(dv1, dv5);
		m2 = _mm512_cmplt_epu64_mask(dv2, dv6);
		m3 = _mm512_cmplt_epu64_mask(dv3, dv7);
		m4 = _mm512_cmplt_epu64_mask(dv4, dv8);

		t1 = _mm512_mask_blend_epi64(m1, dv1, dv5);
		t2 = _mm512_mask_blend_epi64(m2, dv2, dv6);
		t3 = _mm512_mask_blend_epi64(m3, dv3, dv7);
		t4 = _mm512_mask_blend_epi64(m4, dv4, dv8);
		dv5 = _mm512_mask_blend_epi64(m1, dv5, dv1);
		dv6 = _mm512_mask_blend_epi64(m2, dv6, dv2);
		dv7 = _mm512_mask_blend_epi64(m3, dv7, dv3);
		dv8 = _mm512_mask_blend_epi64(m4, dv8, dv4);
		dv1 = t1;
		dv2 = t2;
		dv3 = t3;
		dv4 = t4;
		
		// distance 16 swaps - just compare the vecs
		m1 = _mm512_cmplt_epu64_mask(dv1, dv3);
		m2 = _mm512_cmplt_epu64_mask(dv2, dv4);
		m3 = _mm512_cmplt_epu64_mask(dv5, dv7);
		m4 = _mm512_cmplt_epu64_mask(dv6, dv8);

		t1 = _mm512_mask_blend_epi64(m1, dv1, dv3);
		t2 = _mm512_mask_blend_epi64(m2, dv2, dv4);
		t3 = _mm512_mask_blend_epi64(m3, dv5, dv7);
		t4 = _mm512_mask_blend_epi64(m4, dv6, dv8);

		dv3 = _mm512_mask_blend_epi64(m1, dv3, dv1);
		dv4 = _mm512_mask_blend_epi64(m2, dv4, dv2);
		dv7 = _mm512_mask_blend_epi64(m3, dv7, dv5);
		dv8 = _mm512_mask_blend_epi64(m4, dv8, dv6);

		dv1 = t1;
		dv2 = t2;
		dv5 = t3;
		dv6 = t4;

		// distance 8 swaps - just compare the vecs
		m1 = _mm512_cmplt_epu64_mask(dv1, dv2);
		m2 = _mm512_cmplt_epu64_mask(dv3, dv4);
		m3 = _mm512_cmplt_epu64_mask(dv5, dv6);
		m4 = _mm512_cmplt_epu64_mask(dv7, dv8);

		t1 = _mm512_mask_blend_epi64(m1, dv1, dv2);
		t2 = _mm512_mask_blend_epi64(m2, dv3, dv4);
		t3 = _mm512_mask_blend_epi64(m3, dv5, dv6);
		t4 = _mm512_mask_blend_epi64(m4, dv7, dv8);
		
		dv2 = _mm512_mask_blend_epi64(m1, dv2, dv1);
		dv4 = _mm512_mask_blend_epi64(m2, dv4, dv3);
		dv6 = _mm512_mask_blend_epi64(m3, dv6, dv5);
		dv8 = _mm512_mask_blend_epi64(m4, dv8, dv7);
		
		dv1 = t1;
		dv3 = t2;
		dv5 = t3;
		dv7 = t4;

		// distance 4 swaps
		dv1_swap = SWAP256(dv1);
		dv2_swap = SWAP256(dv2);
		dv3_swap = SWAP256(dv3);
		dv4_swap = SWAP256(dv4);
		dv5_swap = SWAP256(dv5);
		dv6_swap = SWAP256(dv6);
		dv7_swap = SWAP256(dv7);
		dv8_swap = SWAP256(dv8);
		
		m1 = _mm512_cmplt_epu64_mask(dv1, dv1_swap);
		m2 = _mm512_cmplt_epu64_mask(dv2, dv2_swap);
		m3 = _mm512_cmplt_epu64_mask(dv3, dv3_swap);
		m4 = _mm512_cmplt_epu64_mask(dv4, dv4_swap);
		m5 = _mm512_cmplt_epu64_mask(dv5, dv5_swap);
		m6 = _mm512_cmplt_epu64_mask(dv6, dv6_swap);
		m7 = _mm512_cmplt_epu64_mask(dv7, dv7_swap);
		m8 = _mm512_cmplt_epu64_mask(dv8, dv8_swap);

		// 'F0' for the swap between dist-4 lanes, non alternating
		dv1 = _mm512_mask_blend_epi64(m1 ^ 0xF0, dv1, dv1_swap);
		dv2 = _mm512_mask_blend_epi64(m2 ^ 0xF0, dv2, dv2_swap);
		dv3 = _mm512_mask_blend_epi64(m3 ^ 0xF0, dv3, dv3_swap);
		dv4 = _mm512_mask_blend_epi64(m4 ^ 0xF0, dv4, dv4_swap);
		dv5 = _mm512_mask_blend_epi64(m5 ^ 0xF0, dv5, dv5_swap);
		dv6 = _mm512_mask_blend_epi64(m6 ^ 0xF0, dv6, dv6_swap);
		dv7 = _mm512_mask_blend_epi64(m7 ^ 0xF0, dv7, dv7_swap);
		dv8 = _mm512_mask_blend_epi64(m8 ^ 0xF0, dv8, dv8_swap);

		// distance 2 swaps
		dv1_swap = SWAP128(dv1);
		dv2_swap = SWAP128(dv2);
		dv3_swap = SWAP128(dv3);
		dv4_swap = SWAP128(dv4);
		dv5_swap = SWAP128(dv5);
		dv6_swap = SWAP128(dv6);
		dv7_swap = SWAP128(dv7);
		dv8_swap = SWAP128(dv8);
		
		m1 = _mm512_cmplt_epu64_mask(dv1, dv1_swap);
		m2 = _mm512_cmplt_epu64_mask(dv2, dv2_swap);
		m3 = _mm512_cmplt_epu64_mask(dv3, dv3_swap);
		m4 = _mm512_cmplt_epu64_mask(dv4, dv4_swap);
		m5 = _mm512_cmplt_epu64_mask(dv5, dv5_swap);
		m6 = _mm512_cmplt_epu64_mask(dv6, dv6_swap);
		m7 = _mm512_cmplt_epu64_mask(dv7, dv7_swap);
		m8 = _mm512_cmplt_epu64_mask(dv8, dv8_swap);

		// 'CC' for the swap between dist-2 lanes, non alternating
		dv1 = _mm512_mask_blend_epi64(m1 ^ 0xCC, dv1, dv1_swap);
		dv2 = _mm512_mask_blend_epi64(m2 ^ 0xCC, dv2, dv2_swap);
		dv3 = _mm512_mask_blend_epi64(m3 ^ 0xCC, dv3, dv3_swap);
		dv4 = _mm512_mask_blend_epi64(m4 ^ 0xCC, dv4, dv4_swap);
		dv5 = _mm512_mask_blend_epi64(m5 ^ 0xCC, dv5, dv5_swap);
		dv6 = _mm512_mask_blend_epi64(m6 ^ 0xCC, dv6, dv6_swap);
		dv7 = _mm512_mask_blend_epi64(m7 ^ 0xCC, dv7, dv7_swap);
		dv8 = _mm512_mask_blend_epi64(m8 ^ 0xCC, dv8, dv8_swap);

		// adjacent swaps
		dv1_swap = SWAP64(dv1);
		dv2_swap = SWAP64(dv2);
		dv3_swap = SWAP64(dv3);
		dv4_swap = SWAP64(dv4);
		dv5_swap = SWAP64(dv5);
		dv6_swap = SWAP64(dv6);
		dv7_swap = SWAP64(dv7);
		dv8_swap = SWAP64(dv8);
		
		m1 = _mm512_cmplt_epu64_mask(dv1, dv1_swap);
		m2 = _mm512_cmplt_epu64_mask(dv2, dv2_swap);
		m3 = _mm512_cmplt_epu64_mask(dv3, dv3_swap);
		m4 = _mm512_cmplt_epu64_mask(dv4, dv4_swap);
		m5 = _mm512_cmplt_epu64_mask(dv5, dv5_swap);
		m6 = _mm512_cmplt_epu64_mask(dv6, dv6_swap);
		m7 = _mm512_cmplt_epu64_mask(dv7, dv7_swap);
		m8 = _mm512_cmplt_epu64_mask(dv8, dv8_swap);

		// 'AA' for the swap between adjacent lanes
		dv1 = _mm512_mask_blend_epi64(m1 ^ 0xAA, dv1, dv1_swap);
		dv2 = _mm512_mask_blend_epi64(m2 ^ 0xAA, dv2, dv2_swap);
		dv3 = _mm512_mask_blend_epi64(m3 ^ 0xAA, dv3, dv3_swap);
		dv4 = _mm512_mask_blend_epi64(m4 ^ 0xAA, dv4, dv4_swap);
		dv5 = _mm512_mask_blend_epi64(m5 ^ 0xAA, dv5, dv5_swap);
		dv6 = _mm512_mask_blend_epi64(m6 ^ 0xAA, dv6, dv6_swap);
		dv7 = _mm512_mask_blend_epi64(m7 ^ 0xAA, dv7, dv7_swap);
		dv8 = _mm512_mask_blend_epi64(m8 ^ 0xAA, dv8, dv8_swap);

	}
	else
	{

		// distance 32 swaps - just compare the vecs
		m1 = _mm512_cmpgt_epu64_mask(dv1, dv5);
		m2 = _mm512_cmpgt_epu64_mask(dv2, dv6);
		m3 = _mm512_cmpgt_epu64_mask(dv3, dv7);
		m4 = _mm512_cmpgt_epu64_mask(dv4, dv8);

		t1 = _mm512_mask_blend_epi64(m1, dv1, dv5);
		t2 = _mm512_mask_blend_epi64(m2, dv2, dv6);
		t3 = _mm512_mask_blend_epi64(m3, dv3, dv7);
		t4 = _mm512_mask_blend_epi64(m4, dv4, dv8);
		dv5 = _mm512_mask_blend_epi64(m1, dv5, dv1);
		dv6 = _mm512_mask_blend_epi64(m2, dv6, dv2);
		dv7 = _mm512_mask_blend_epi64(m3, dv7, dv3);
		dv8 = _mm512_mask_blend_epi64(m4, dv8, dv4);
		dv1 = t1;
		dv2 = t2;
		dv3 = t3;
		dv4 = t4;
		
		// distance 16 swaps - just compare the vecs
		m1 = _mm512_cmpgt_epu64_mask(dv1, dv3);
		m2 = _mm512_cmpgt_epu64_mask(dv2, dv4);
		m3 = _mm512_cmpgt_epu64_mask(dv5, dv7);
		m4 = _mm512_cmpgt_epu64_mask(dv6, dv8);

		t1 = _mm512_mask_blend_epi64(m1, dv1, dv3);
		t2 = _mm512_mask_blend_epi64(m2, dv2, dv4);
		t3 = _mm512_mask_blend_epi64(m3, dv5, dv7);
		t4 = _mm512_mask_blend_epi64(m4, dv6, dv8);

		dv3 = _mm512_mask_blend_epi64(m1, dv3, dv1);
		dv4 = _mm512_mask_blend_epi64(m2, dv4, dv2);
		dv7 = _mm512_mask_blend_epi64(m3, dv7, dv5);
		dv8 = _mm512_mask_blend_epi64(m4, dv8, dv6);

		dv1 = t1;
		dv2 = t2;
		dv5 = t3;
		dv6 = t4;

		// distance 8 swaps - just compare the vecs
		m1 = _mm512_cmpgt_epu64_mask(dv1, dv2);
		m2 = _mm512_cmpgt_epu64_mask(dv3, dv4);
		m3 = _mm512_cmpgt_epu64_mask(dv5, dv6);
		m4 = _mm512_cmpgt_epu64_mask(dv7, dv8);

		t1 = _mm512_mask_blend_epi64(m1, dv1, dv2);
		t2 = _mm512_mask_blend_epi64(m2, dv3, dv4);
		t3 = _mm512_mask_blend_epi64(m3, dv5, dv6);
		t4 = _mm512_mask_blend_epi64(m4, dv7, dv8);
		
		dv2 = _mm512_mask_blend_epi64(m1, dv2, dv1);
		dv4 = _mm512_mask_blend_epi64(m2, dv4, dv3);
		dv6 = _mm512_mask_blend_epi64(m3, dv6, dv5);
		dv8 = _mm512_mask_blend_epi64(m4, dv8, dv7);
		
		dv1 = t1;
		dv3 = t2;
		dv5 = t3;
		dv7 = t4;

		// distance 4 swaps
		dv1_swap = SWAP256(dv1);
		dv2_swap = SWAP256(dv2);
		dv3_swap = SWAP256(dv3);
		dv4_swap = SWAP256(dv4);
		dv5_swap = SWAP256(dv5);
		dv6_swap = SWAP256(dv6);
		dv7_swap = SWAP256(dv7);
		dv8_swap = SWAP256(dv8);
		
		m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
		m2 = _mm512_cmpgt_epu64_mask(dv2, dv2_swap);
		m3 = _mm512_cmpgt_epu64_mask(dv3, dv3_swap);
		m4 = _mm512_cmpgt_epu64_mask(dv4, dv4_swap);
		m5 = _mm512_cmpgt_epu64_mask(dv5, dv5_swap);
		m6 = _mm512_cmpgt_epu64_mask(dv6, dv6_swap);
		m7 = _mm512_cmpgt_epu64_mask(dv7, dv7_swap);
		m8 = _mm512_cmpgt_epu64_mask(dv8, dv8_swap);

		// 'F0' for the swap between dist-4 lanes, non alternating
		dv1 = _mm512_mask_blend_epi64(m1 ^ 0xF0, dv1, dv1_swap);
		dv2 = _mm512_mask_blend_epi64(m2 ^ 0xF0, dv2, dv2_swap);
		dv3 = _mm512_mask_blend_epi64(m3 ^ 0xF0, dv3, dv3_swap);
		dv4 = _mm512_mask_blend_epi64(m4 ^ 0xF0, dv4, dv4_swap);
		dv5 = _mm512_mask_blend_epi64(m5 ^ 0xF0, dv5, dv5_swap);
		dv6 = _mm512_mask_blend_epi64(m6 ^ 0xF0, dv6, dv6_swap);
		dv7 = _mm512_mask_blend_epi64(m7 ^ 0xF0, dv7, dv7_swap);
		dv8 = _mm512_mask_blend_epi64(m8 ^ 0xF0, dv8, dv8_swap);

		// distance 2 swaps
		dv1_swap = SWAP128(dv1);
		dv2_swap = SWAP128(dv2);
		dv3_swap = SWAP128(dv3);
		dv4_swap = SWAP128(dv4);
		dv5_swap = SWAP128(dv5);
		dv6_swap = SWAP128(dv6);
		dv7_swap = SWAP128(dv7);
		dv8_swap = SWAP128(dv8);
		
		m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
		m2 = _mm512_cmpgt_epu64_mask(dv2, dv2_swap);
		m3 = _mm512_cmpgt_epu64_mask(dv3, dv3_swap);
		m4 = _mm512_cmpgt_epu64_mask(dv4, dv4_swap);
		m5 = _mm512_cmpgt_epu64_mask(dv5, dv5_swap);
		m6 = _mm512_cmpgt_epu64_mask(dv6, dv6_swap);
		m7 = _mm512_cmpgt_epu64_mask(dv7, dv7_swap);
		m8 = _mm512_cmpgt_epu64_mask(dv8, dv8_swap);

		// 'CC' for the swap between dist-2 lanes, non alternating
		dv1 = _mm512_mask_blend_epi64(m1 ^ 0xCC, dv1, dv1_swap);
		dv2 = _mm512_mask_blend_epi64(m2 ^ 0xCC, dv2, dv2_swap);
		dv3 = _mm512_mask_blend_epi64(m3 ^ 0xCC, dv3, dv3_swap);
		dv4 = _mm512_mask_blend_epi64(m4 ^ 0xCC, dv4, dv4_swap);
		dv5 = _mm512_mask_blend_epi64(m5 ^ 0xCC, dv5, dv5_swap);
		dv6 = _mm512_mask_blend_epi64(m6 ^ 0xCC, dv6, dv6_swap);
		dv7 = _mm512_mask_blend_epi64(m7 ^ 0xCC, dv7, dv7_swap);
		dv8 = _mm512_mask_blend_epi64(m8 ^ 0xCC, dv8, dv8_swap);

		// adjacent swaps
		dv1_swap = SWAP64(dv1);
		dv2_swap = SWAP64(dv2);
		dv3_swap = SWAP64(dv3);
		dv4_swap = SWAP64(dv4);
		dv5_swap = SWAP64(dv5);
		dv6_swap = SWAP64(dv6);
		dv7_swap = SWAP64(dv7);
		dv8_swap = SWAP64(dv8);
		
		m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
		m2 = _mm512_cmpgt_epu64_mask(dv2, dv2_swap);
		m3 = _mm512_cmpgt_epu64_mask(dv3, dv3_swap);
		m4 = _mm512_cmpgt_epu64_mask(dv4, dv4_swap);
		m5 = _mm512_cmpgt_epu64_mask(dv5, dv5_swap);
		m6 = _mm512_cmpgt_epu64_mask(dv6, dv6_swap);
		m7 = _mm512_cmpgt_epu64_mask(dv7, dv7_swap);
		m8 = _mm512_cmpgt_epu64_mask(dv8, dv8_swap);

		// 'AA' for the swap between adjacent lanes
		dv1 = _mm512_mask_blend_epi64(m1 ^ 0xAA, dv1, dv1_swap);
		dv2 = _mm512_mask_blend_epi64(m2 ^ 0xAA, dv2, dv2_swap);
		dv3 = _mm512_mask_blend_epi64(m3 ^ 0xAA, dv3, dv3_swap);
		dv4 = _mm512_mask_blend_epi64(m4 ^ 0xAA, dv4, dv4_swap);
		dv5 = _mm512_mask_blend_epi64(m5 ^ 0xAA, dv5, dv5_swap);
		dv6 = _mm512_mask_blend_epi64(m6 ^ 0xAA, dv6, dv6_swap);
		dv7 = _mm512_mask_blend_epi64(m7 ^ 0xAA, dv7, dv7_swap);
		dv8 = _mm512_mask_blend_epi64(m8 ^ 0xAA, dv8, dv8_swap);
	}

	_mm512_store_epi64(data, dv1);
	_mm512_store_epi64(data + 8, dv2);
	_mm512_store_epi64(data + 16, dv3);
	_mm512_store_epi64(data + 24, dv4);
	_mm512_store_epi64(data + 32, dv5);
	_mm512_store_epi64(data + 40, dv6);
	_mm512_store_epi64(data + 48, dv7);
	_mm512_store_epi64(data + 56, dv8);

	return;
}

// on Epyc, max/min_epu64 is slower: the instruction is 
// latency 3, throughput 1 instead of 1/0.5 for epu32
void bitonic_sort_dir_64_minmax(uint64_t* data, int dir) 
{
	// sort 64 64-bit elements
	int i, j;

	__m512i t1;
	__m512i t2;
	__m512i t3;
	__m512i t4;
	__m512i t5;
	__m512i t6;
	__m512i t7;
	__m512i t8;
	__m512i dv1;
	__m512i dv2;
	__m512i dv3;
	__m512i dv4;
	__m512i dv5;
	__m512i dv6;
	__m512i dv7;
	__m512i dv8;
	__m512i dv1_swap;
	__m512i dv2_swap;
	__m512i dv3_swap;
	__m512i dv4_swap;
	__m512i dv5_swap;
	__m512i dv6_swap;
	__m512i dv7_swap;
	__m512i dv8_swap;
	__mmask16 m1;
	__mmask16 m2;
	__mmask16 m3;
	__mmask16 m4;
	__mmask16 m5;
	__mmask16 m6;
	__mmask16 m7;
	__mmask16 m8;

	dv1 = _mm512_load_epi64(data);
	dv2 = _mm512_load_epi64(data + 8);
	dv3 = _mm512_load_epi64(data + 16);
	dv4 = _mm512_load_epi64(data + 24);
	dv5 = _mm512_load_epi64(data + 32);
	dv6 = _mm512_load_epi64(data + 40);
	dv7 = _mm512_load_epi64(data + 48);
	dv8 = _mm512_load_epi64(data + 56);

	// phase 1: dist-2 alternating compares ('CC')
	
	// adjacent swaps, alternating compares
	dv1_swap = SWAP64(dv1);
	dv2_swap = SWAP64(dv2);
	dv3_swap = SWAP64(dv3);
	dv4_swap = SWAP64(dv4);
	dv5_swap = SWAP64(dv5);
	dv6_swap = SWAP64(dv6);
	dv7_swap = SWAP64(dv7);
	dv8_swap = SWAP64(dv8);
	
	// 'AA' for the swap between adjacent lanes ^ 'CC' to alternate gt vs. le --> 0x66
	MINMAX64(0x66, 0x99, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8)
	
	// phase 2: dist-4 alternating compares ('F0')
	
	// distance 2 swaps
	dv1_swap = SWAP128(dv1);
	dv2_swap = SWAP128(dv2);
	dv3_swap = SWAP128(dv3);
	dv4_swap = SWAP128(dv4);
	dv5_swap = SWAP128(dv5);
	dv6_swap = SWAP128(dv6);
	dv7_swap = SWAP128(dv7);
	dv8_swap = SWAP128(dv8);

	// 'CC' for the swap between dist-2 lanes ^ 'F0' to alternate gt vs. le --> 0x3C
	MINMAX64(0x3C, 0xC3, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8)

	// adjacent swaps
	dv1_swap = SWAP64(dv1);
	dv2_swap = SWAP64(dv2);
	dv3_swap = SWAP64(dv3);
	dv4_swap = SWAP64(dv4);
	dv5_swap = SWAP64(dv5);
	dv6_swap = SWAP64(dv6);
	dv7_swap = SWAP64(dv7);
	dv8_swap = SWAP64(dv8);

	// 'AA' for the swap between adjacent lanes ^ 'F0' to alternate gt vs. le --> 0x5A
	MINMAX64(0x5A, 0xA5, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	// phase 3: dist-8 alternating compares ('FF00')

	// distance 4 swaps
	dv1_swap = SWAP256(dv1);
	dv2_swap = SWAP256(dv2);
	dv3_swap = SWAP256(dv3);
	dv4_swap = SWAP256(dv4);
	dv5_swap = SWAP256(dv5);
	dv6_swap = SWAP256(dv6);
	dv7_swap = SWAP256(dv7);
	dv8_swap = SWAP256(dv8);

	// 'F0F0' for the swap between dist-4 lanes ^ 'FF00' to alternate gt vs. le --> 0x0FF0
	MINMAX64_alt1(0xF0, 0x0f, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	// distance 2 swaps
	dv1_swap = SWAP128(dv1);
	dv2_swap = SWAP128(dv2);
	dv3_swap = SWAP128(dv3);
	dv4_swap = SWAP128(dv4);
	dv5_swap = SWAP128(dv5);
	dv6_swap = SWAP128(dv6);
	dv7_swap = SWAP128(dv7);
	dv8_swap = SWAP128(dv8);

	// 'CCCC' for the swap between dist-2 lanes ^ 'FF00' to alternate gt vs. le --> 0x33CC
	MINMAX64_alt1(0xCC, 0x33, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	// adjacent swaps
	dv1_swap = SWAP64(dv1);
	dv2_swap = SWAP64(dv2);
	dv3_swap = SWAP64(dv3);
	dv4_swap = SWAP64(dv4);
	dv5_swap = SWAP64(dv5);
	dv6_swap = SWAP64(dv6);
	dv7_swap = SWAP64(dv7);
	dv8_swap = SWAP64(dv8);

	// 'AA' for the swap between adjacent lanes ^ 'FF00' to alternate gt vs. le --> 0x55AA
	MINMAX64_alt1(0xAA, 0x55, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	// phase 4: dist-16 alternating compares (alternating gt/lt)
	
	// distance 8 swaps (256 bits)
	t1 =  _mm512_min_epu64(dv1, dv2);
	t2 =  _mm512_max_epu64(dv3, dv4);
	t3 =  _mm512_min_epu64(dv5, dv6);
	t4 =  _mm512_max_epu64(dv7, dv8);
	dv2 = _mm512_max_epu64(dv1, dv2);
	dv4 = _mm512_min_epu64(dv3, dv4);
	dv6 = _mm512_max_epu64(dv5, dv6);
	dv8 = _mm512_min_epu64(dv7, dv8);
	dv1 = t1;
	dv3 = t2;
	dv5 = t3;
	dv7 = t4;


	// distance 4 swaps
	dv1_swap = SWAP256(dv1);
	dv2_swap = SWAP256(dv2);
	dv3_swap = SWAP256(dv3);
	dv4_swap = SWAP256(dv4);
	dv5_swap = SWAP256(dv5);
	dv6_swap = SWAP256(dv6);
	dv7_swap = SWAP256(dv7);
	dv8_swap = SWAP256(dv8);

	// 'F0' for the swap between dist-4 lanes, non alternating
	MINMAX64_alt2(0xF0, 0x0f, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	// distance 2 swaps
	dv1_swap = SWAP128(dv1);
	dv2_swap = SWAP128(dv2);
	dv3_swap = SWAP128(dv3);
	dv4_swap = SWAP128(dv4);
	dv5_swap = SWAP128(dv5);
	dv6_swap = SWAP128(dv6);
	dv7_swap = SWAP128(dv7);
	dv8_swap = SWAP128(dv8);

	// 'CC' for the swap between dist-2 lanes, non alternating
	MINMAX64_alt2(0xCC, 0x33, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	// adjacent swaps
	dv1_swap = SWAP64(dv1);
	dv2_swap = SWAP64(dv2);
	dv3_swap = SWAP64(dv3);
	dv4_swap = SWAP64(dv4);
	dv5_swap = SWAP64(dv5);
	dv6_swap = SWAP64(dv6);
	dv7_swap = SWAP64(dv7);
	dv8_swap = SWAP64(dv8);

	// 'AA' for the swap between adjacent lanes, non alternating
	MINMAX64_alt2(0xAA, 0x55, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	// phase 5: dist-32 alternating compares (alternating gtgt/ltlt)

	// distance 16 swaps - just compare the vecs
	t1 =  _mm512_min_epu64(dv1, dv3);
	t2 =  _mm512_min_epu64(dv2, dv4);
	t3 =  _mm512_max_epu64(dv5, dv7);
	t4 =  _mm512_max_epu64(dv6, dv8);
	dv3 = _mm512_max_epu64(dv1, dv3);
	dv4 = _mm512_max_epu64(dv2, dv4);
	dv7 = _mm512_min_epu64(dv5, dv7);
	dv8 = _mm512_min_epu64(dv6, dv8);
	dv1 = t1;
	dv2 = t2;
	dv5 = t3;
	dv6 = t4;

	// distance 8 swaps (256 bits)
	t1 =  _mm512_min_epu64(dv1, dv2);
	t2 =  _mm512_min_epu64(dv3, dv4);
	t3 =  _mm512_max_epu64(dv5, dv6);
	t4 =  _mm512_max_epu64(dv7, dv8);
	dv2 = _mm512_max_epu64(dv1, dv2);
	dv4 = _mm512_max_epu64(dv3, dv4);
	dv6 = _mm512_min_epu64(dv5, dv6);
	dv8 = _mm512_min_epu64(dv7, dv8);
	dv1 = t1;
	dv3 = t2;
	dv5 = t3;
	dv7 = t4;

	// distance 4 swaps
	dv1_swap = SWAP256(dv1);
	dv2_swap = SWAP256(dv2);
	dv3_swap = SWAP256(dv3);
	dv4_swap = SWAP256(dv4);
	dv5_swap = SWAP256(dv5);
	dv6_swap = SWAP256(dv6);
	dv7_swap = SWAP256(dv7);
	dv8_swap = SWAP256(dv8);

	// 'F0' for the swap between dist-4 lanes, non alternating
	MINMAX64_alt4(0xF0, 0x0f, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	// distance 2 swaps
	dv1_swap = SWAP128(dv1);
	dv2_swap = SWAP128(dv2);
	dv3_swap = SWAP128(dv3);
	dv4_swap = SWAP128(dv4);
	dv5_swap = SWAP128(dv5);
	dv6_swap = SWAP128(dv6);
	dv7_swap = SWAP128(dv7);
	dv8_swap = SWAP128(dv8);

	// 'CC' for the swap between dist-2 lanes, non alternating
	MINMAX64_alt4(0xCC, 0x33, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	// adjacent swaps
	dv1_swap = SWAP64(dv1);
	dv2_swap = SWAP64(dv2);
	dv3_swap = SWAP64(dv3);
	dv4_swap = SWAP64(dv4);
	dv5_swap = SWAP64(dv5);
	dv6_swap = SWAP64(dv6);
	dv7_swap = SWAP64(dv7);
	dv8_swap = SWAP64(dv8);

	// 'AA' for the swap between adjacent lanes
	MINMAX64_alt4(0xAA, 0x55, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
	


	// phase 6: merge (all same compare 'dir')
	if (dir == 1)
	{	
		// distance 32 swaps - just compare the vecs
		t1 =  _mm512_max_epu64(dv1, dv5);
		t2 =  _mm512_max_epu64(dv2, dv6);
		t3 =  _mm512_max_epu64(dv3, dv7);
		t4 =  _mm512_max_epu64(dv4, dv8);
		dv5 = _mm512_min_epu64(dv1, dv5);
		dv6 = _mm512_min_epu64(dv2, dv6);
		dv7 = _mm512_min_epu64(dv3, dv7);
		dv8 = _mm512_min_epu64(dv4, dv8);
		dv1 = t1;
		dv2 = t2;
		dv3 = t3;
		dv4 = t4;

		// distance 16 swaps - just compare the vecs
		t1 =  _mm512_max_epu64(dv1, dv3);
		t2 =  _mm512_max_epu64(dv2, dv4);
		t3 =  _mm512_max_epu64(dv5, dv7);
		t4 =  _mm512_max_epu64(dv6, dv8);
		dv3 = _mm512_min_epu64(dv1, dv3);
		dv4 = _mm512_min_epu64(dv2, dv4);
		dv7 = _mm512_min_epu64(dv5, dv7);
		dv8 = _mm512_min_epu64(dv6, dv8);
		dv1 = t1;
		dv2 = t2;
		dv5 = t3;
		dv6 = t4;
		
		// distance 8 swaps - 256 bits
		t1 =  _mm512_max_epu64(dv1, dv2);
		t2 =  _mm512_max_epu64(dv3, dv4);
		t3 =  _mm512_max_epu64(dv5, dv6);
		t4 =  _mm512_max_epu64(dv7, dv8);
		dv2 = _mm512_min_epu64(dv1, dv2);
		dv4 = _mm512_min_epu64(dv3, dv4);
		dv6 = _mm512_min_epu64(dv5, dv6);
		dv8 = _mm512_min_epu64(dv7, dv8);
		dv1 = t1;
		dv3 = t2;
		dv5 = t3;
		dv7 = t4;
		
		// distance 4 swaps
		dv1_swap = SWAP256(dv1);
		dv2_swap = SWAP256(dv2);
		dv3_swap = SWAP256(dv3);
		dv4_swap = SWAP256(dv4);
		dv5_swap = SWAP256(dv5);
		dv6_swap = SWAP256(dv6);
		dv7_swap = SWAP256(dv7);
		dv8_swap = SWAP256(dv8);

		// 'F0' for the swap between dist-4 lanes, non alternating
		MINMAX64(0x0f, 0xF0, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		
		// distance 2 swaps
		dv1_swap = SWAP128(dv1);
		dv2_swap = SWAP128(dv2);
		dv3_swap = SWAP128(dv3);
		dv4_swap = SWAP128(dv4);
		dv5_swap = SWAP128(dv5);
		dv6_swap = SWAP128(dv6);
		dv7_swap = SWAP128(dv7);
		dv8_swap = SWAP128(dv8);

		// 'CC' for the swap between dist-2 lanes, non alternating
		MINMAX64(0x33, 0xCC, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		
		// adjacent swaps
		dv1_swap = SWAP64(dv1);
		dv2_swap = SWAP64(dv2);
		dv3_swap = SWAP64(dv3);
		dv4_swap = SWAP64(dv4);
		dv5_swap = SWAP64(dv5);
		dv6_swap = SWAP64(dv6);
		dv7_swap = SWAP64(dv7);
		dv8_swap = SWAP64(dv8);

		// 'AA' for the swap between adjacent lanes
		MINMAX64(0x55, 0xaa, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	}
	else
	{	
		// distance 32 swaps - just compare the vecs
		t1 =  _mm512_min_epu64(dv1, dv5);
		t2 =  _mm512_min_epu64(dv2, dv6);
		t3 =  _mm512_min_epu64(dv3, dv7);
		t4 =  _mm512_min_epu64(dv4, dv8);
		dv5 = _mm512_max_epu64(dv1, dv5);
		dv6 = _mm512_max_epu64(dv2, dv6);
		dv7 = _mm512_max_epu64(dv3, dv7);
		dv8 = _mm512_max_epu64(dv4, dv8);
		dv1 = t1;
		dv2 = t2;
		dv3 = t3;
		dv4 = t4;

		// distance 16 swaps - just compare the vecs
		t1 =  _mm512_min_epu64(dv1, dv3);
		t2 =  _mm512_min_epu64(dv2, dv4);
		t3 =  _mm512_min_epu64(dv5, dv7);
		t4 =  _mm512_min_epu64(dv6, dv8);
		dv3 = _mm512_max_epu64(dv1, dv3);
		dv4 = _mm512_max_epu64(dv2, dv4);
		dv7 = _mm512_max_epu64(dv5, dv7);
		dv8 = _mm512_max_epu64(dv6, dv8);
		dv1 = t1;
		dv2 = t2;
		dv5 = t3;
		dv6 = t4;

		// distance 8 swaps - 256 bits
		t1 =  _mm512_min_epu64(dv1, dv2);
		t2 =  _mm512_min_epu64(dv3, dv4);
		t3 =  _mm512_min_epu64(dv5, dv6);
		t4 =  _mm512_min_epu64(dv7, dv8);
		dv2 = _mm512_max_epu64(dv1, dv2);
		dv4 = _mm512_max_epu64(dv3, dv4);
		dv6 = _mm512_max_epu64(dv5, dv6);
		dv8 = _mm512_max_epu64(dv7, dv8);
		dv1 = t1;
		dv3 = t2;
		dv5 = t3;
		dv7 = t4;
		
		// distance 4 swaps
		dv1_swap = SWAP256(dv1);
		dv2_swap = SWAP256(dv2);
		dv3_swap = SWAP256(dv3);
		dv4_swap = SWAP256(dv4);
		dv5_swap = SWAP256(dv5);
		dv6_swap = SWAP256(dv6);
		dv7_swap = SWAP256(dv7);
		dv8_swap = SWAP256(dv8);

		// '0F' for the swap between dist-4 lanes, non alternating
		MINMAX64(0xf0, 0x0f, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

		// distance 2 swaps
		dv1_swap = SWAP128(dv1);
		dv2_swap = SWAP128(dv2);
		dv3_swap = SWAP128(dv3);
		dv4_swap = SWAP128(dv4);
		dv5_swap = SWAP128(dv5);
		dv6_swap = SWAP128(dv6);
		dv7_swap = SWAP128(dv7);
		dv8_swap = SWAP128(dv8);

		// 'CC' for the swap between dist-2 lanes, non alternating
		MINMAX64(0xCC, 0x33, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);
		
		// adjacent swaps
		dv1_swap = SWAP64(dv1);
		dv2_swap = SWAP64(dv2);
		dv3_swap = SWAP64(dv3);
		dv4_swap = SWAP64(dv4);
		dv5_swap = SWAP64(dv5);
		dv6_swap = SWAP64(dv6);
		dv7_swap = SWAP64(dv7);
		dv8_swap = SWAP64(dv8);

		// 'AA' for the swap between adjacent lanes
		MINMAX64(0xaa, 0x55, dv1, dv2, dv3, dv4, dv5, dv6, dv7, dv8);

	}

	_mm512_store_epi64(data, dv1);
	_mm512_store_epi64(data + 8, dv2);
	_mm512_store_epi64(data + 16, dv3);
	_mm512_store_epi64(data + 24, dv4);
	_mm512_store_epi64(data + 32, dv5);
	_mm512_store_epi64(data + 40, dv6);
	_mm512_store_epi64(data + 48, dv7);
	_mm512_store_epi64(data + 56, dv8);
	
	return;
}

void bitonic_merge(uint64_t *data, uint32_t sz, int dir)
{
	if (sz <= 64)
	{
		// base case: do the hardcoded 64-element sort
		bitonic_merge_dir_64(data, dir);
		return;
	}
	
	// half-size cmp/swap
	int i;

	// we have sz/2 swaps to do at a stride of sz/2.
	// the number of swaps will be divisible by 64 because
	// sz is at least 128 (basecase is 64).
	// can batch up the swaps:

	if (dir == 1)
	{
		// 128-element merge passes at a stride of sz/2
		for (i = 0; i < sz / 128; i++)
		{
			__m512i t1;
			__m512i t2;
			__m512i t3;
			__m512i t4;
			__m512i t5;
			__m512i t6;
			__m512i t7;
			__m512i t8;
			__m512i dv1;
			__m512i dv2;
			__m512i dv3;
			__m512i dv4;
			__m512i dv5;
			__m512i dv6;
			__m512i dv7;
			__m512i dv8;
			__m512i dv9;
			__m512i dv10;
			__m512i dv11;
			__m512i dv12;
			__m512i dv13;
			__m512i dv14;
			__m512i dv15;
			__m512i dv16;

			__mmask8 m1;
			__mmask8 m2;
			__mmask8 m3;
			__mmask8 m4;
			__mmask8 m5;
			__mmask8 m6;
			__mmask8 m7;
			__mmask8 m8;

			// swap 64 elements at starting offset i * 64
			dv1 = _mm512_load_epi64(data + i * 64 + 0);
			dv2 = _mm512_load_epi64(data + i * 64 + 8);
			dv3 = _mm512_load_epi64(data + i * 64 + 16);
			dv4 = _mm512_load_epi64(data + i * 64 + 24);
			dv5 = _mm512_load_epi64(data + i * 64 + 32);
			dv6 = _mm512_load_epi64(data + i * 64 + 40);
			dv7 = _mm512_load_epi64(data + i * 64 + 48);
			dv8 = _mm512_load_epi64(data + i * 64 + 56);
			
			// with 64 elements at offset i * 64 + stride (sz/2)
			dv9 = _mm512_load_epi64(data  + i * 64 + sz/2 + 0);
			dv10 = _mm512_load_epi64(data + i * 64 + sz/2 + 8);
			dv11 = _mm512_load_epi64(data + i * 64 + sz/2 + 16);
			dv12 = _mm512_load_epi64(data + i * 64 + sz/2 + 24);
			dv13 = _mm512_load_epi64(data + i * 64 + sz/2 + 32);
			dv14 = _mm512_load_epi64(data + i * 64 + sz/2 + 40);
			dv15 = _mm512_load_epi64(data + i * 64 + sz/2 + 48);
			dv16 = _mm512_load_epi64(data + i * 64 + sz/2 + 56);

				m1 = _mm512_cmplt_epu64_mask(dv1, dv9);
				m2 = _mm512_cmplt_epu64_mask(dv2, dv10);
				m3 = _mm512_cmplt_epu64_mask(dv3, dv11);
				m4 = _mm512_cmplt_epu64_mask(dv4, dv12);
				m5 = _mm512_cmplt_epu64_mask(dv5, dv13);
				m6 = _mm512_cmplt_epu64_mask(dv6, dv14);
				m7 = _mm512_cmplt_epu64_mask(dv7, dv15);
				m8 = _mm512_cmplt_epu64_mask(dv8, dv16);


			t1 = _mm512_mask_blend_epi64(m1, dv1, dv9);
			t2 = _mm512_mask_blend_epi64(m2, dv2, dv10);
			t3 = _mm512_mask_blend_epi64(m3, dv3, dv11);
			t4 = _mm512_mask_blend_epi64(m4, dv4, dv12);
			t5 = _mm512_mask_blend_epi64(m5, dv5, dv13);
			t6 = _mm512_mask_blend_epi64(m6, dv6, dv14);
			t7 = _mm512_mask_blend_epi64(m7, dv7, dv15);
			t8 = _mm512_mask_blend_epi64(m8, dv8, dv16);
			
			dv9 = _mm512_mask_blend_epi64(m1, dv9, dv1);
			dv10 = _mm512_mask_blend_epi64(m2, dv10, dv2);
			dv11 = _mm512_mask_blend_epi64(m3, dv11, dv3);
			dv12 = _mm512_mask_blend_epi64(m4, dv12, dv4);
			dv13 = _mm512_mask_blend_epi64(m5, dv13, dv5);
			dv14 = _mm512_mask_blend_epi64(m6, dv14, dv6);
			dv15 = _mm512_mask_blend_epi64(m7, dv15, dv7);
			dv16 = _mm512_mask_blend_epi64(m8, dv16, dv8);

			dv1 = t1;
			dv2 = t2;
			dv3 = t3;
			dv4 = t4;
			dv5 = t5;
			dv6 = t6;
			dv7 = t7;
			dv8 = t8;
			
			_mm512_store_epi64(data + i * 64 + 0, dv1);
			_mm512_store_epi64(data + i * 64 + 8, dv2);
			_mm512_store_epi64(data + i * 64 + 16, dv3);
			_mm512_store_epi64(data + i * 64 + 24, dv4);
			_mm512_store_epi64(data + i * 64 + 32, dv5);
			_mm512_store_epi64(data + i * 64 + 40, dv6);
			_mm512_store_epi64(data + i * 64 + 48, dv7);
			_mm512_store_epi64(data + i * 64 + 56, dv8);
			
			_mm512_store_epi64(data + i * 64 + sz/2 + 0,  dv9);
			_mm512_store_epi64(data + i * 64 + sz/2 + 8,  dv10);
			_mm512_store_epi64(data + i * 64 + sz/2 + 16, dv11);
			_mm512_store_epi64(data + i * 64 + sz/2 + 24, dv12);
			_mm512_store_epi64(data + i * 64 + sz/2 + 32, dv13);
			_mm512_store_epi64(data + i * 64 + sz/2 + 40, dv14);
			_mm512_store_epi64(data + i * 64 + sz/2 + 48, dv15);
			_mm512_store_epi64(data + i * 64 + sz/2 + 56, dv16);

		}
	}
	else
	{
		// 128-element merge passes at a stride of sz/2
		for (i = 0; i < sz / 128; i++)
		{
			__m512i t1;
			__m512i t2;
			__m512i t3;
			__m512i t4;
			__m512i t5;
			__m512i t6;
			__m512i t7;
			__m512i t8;
			__m512i dv1;
			__m512i dv2;
			__m512i dv3;
			__m512i dv4;
			__m512i dv5;
			__m512i dv6;
			__m512i dv7;
			__m512i dv8;
			__m512i dv9;
			__m512i dv10;
			__m512i dv11;
			__m512i dv12;
			__m512i dv13;
			__m512i dv14;
			__m512i dv15;
			__m512i dv16;

			__mmask8 m1;
			__mmask8 m2;
			__mmask8 m3;
			__mmask8 m4;
			__mmask8 m5;
			__mmask8 m6;
			__mmask8 m7;
			__mmask8 m8;

			// swap 64 elements at starting offset i * 64
			dv1 = _mm512_load_epi64(data + i * 64 + 0);
			dv2 = _mm512_load_epi64(data + i * 64 + 8);
			dv3 = _mm512_load_epi64(data + i * 64 + 16);
			dv4 = _mm512_load_epi64(data + i * 64 + 24);
			dv5 = _mm512_load_epi64(data + i * 64 + 32);
			dv6 = _mm512_load_epi64(data + i * 64 + 40);
			dv7 = _mm512_load_epi64(data + i * 64 + 48);
			dv8 = _mm512_load_epi64(data + i * 64 + 56);
		
			// with 64 elements at offset i * 64 + stride (sz/2)
			dv9 = _mm512_load_epi64(data  + i * 64 + sz/2 + 0);
			dv10 = _mm512_load_epi64(data + i * 64 + sz/2 + 8);
			dv11 = _mm512_load_epi64(data + i * 64 + sz/2 + 16);
			dv12 = _mm512_load_epi64(data + i * 64 + sz/2 + 24);
			dv13 = _mm512_load_epi64(data + i * 64 + sz/2 + 32);
			dv14 = _mm512_load_epi64(data + i * 64 + sz/2 + 40);
			dv15 = _mm512_load_epi64(data + i * 64 + sz/2 + 48);
			dv16 = _mm512_load_epi64(data + i * 64 + sz/2 + 56);

				m1 = _mm512_cmpgt_epu64_mask(dv1, dv9);
				m2 = _mm512_cmpgt_epu64_mask(dv2, dv10);
				m3 = _mm512_cmpgt_epu64_mask(dv3, dv11);
				m4 = _mm512_cmpgt_epu64_mask(dv4, dv12);
				m5 = _mm512_cmpgt_epu64_mask(dv5, dv13);
				m6 = _mm512_cmpgt_epu64_mask(dv6, dv14);
				m7 = _mm512_cmpgt_epu64_mask(dv7, dv15);
				m8 = _mm512_cmpgt_epu64_mask(dv8, dv16);

			t1 = _mm512_mask_blend_epi64(m1, dv1, dv9);
			t2 = _mm512_mask_blend_epi64(m2, dv2, dv10);
			t3 = _mm512_mask_blend_epi64(m3, dv3, dv11);
			t4 = _mm512_mask_blend_epi64(m4, dv4, dv12);
			t5 = _mm512_mask_blend_epi64(m5, dv5, dv13);
			t6 = _mm512_mask_blend_epi64(m6, dv6, dv14);
			t7 = _mm512_mask_blend_epi64(m7, dv7, dv15);
			t8 = _mm512_mask_blend_epi64(m8, dv8, dv16);
			
			dv9 = _mm512_mask_blend_epi64(m1, dv9, dv1);
			dv10 = _mm512_mask_blend_epi64(m2, dv10, dv2);
			dv11 = _mm512_mask_blend_epi64(m3, dv11, dv3);
			dv12 = _mm512_mask_blend_epi64(m4, dv12, dv4);
			dv13 = _mm512_mask_blend_epi64(m5, dv13, dv5);
			dv14 = _mm512_mask_blend_epi64(m6, dv14, dv6);
			dv15 = _mm512_mask_blend_epi64(m7, dv15, dv7);
			dv16 = _mm512_mask_blend_epi64(m8, dv16, dv8);

			dv1 = t1;
			dv2 = t2;
			dv3 = t3;
			dv4 = t4;
			dv5 = t5;
			dv6 = t6;
			dv7 = t7;
			dv8 = t8;
			
			_mm512_store_epi64(data + i * 64 + 0, dv1);
			_mm512_store_epi64(data + i * 64 + 8, dv2);
			_mm512_store_epi64(data + i * 64 + 16, dv3);
			_mm512_store_epi64(data + i * 64 + 24, dv4);
			_mm512_store_epi64(data + i * 64 + 32, dv5);
			_mm512_store_epi64(data + i * 64 + 40, dv6);
			_mm512_store_epi64(data + i * 64 + 48, dv7);
			_mm512_store_epi64(data + i * 64 + 56, dv8);
			
			_mm512_store_epi64(data + i * 64 + sz/2 + 0,  dv9);
			_mm512_store_epi64(data + i * 64 + sz/2 + 8,  dv10);
			_mm512_store_epi64(data + i * 64 + sz/2 + 16, dv11);
			_mm512_store_epi64(data + i * 64 + sz/2 + 24, dv12);
			_mm512_store_epi64(data + i * 64 + sz/2 + 32, dv13);
			_mm512_store_epi64(data + i * 64 + sz/2 + 40, dv14);
			_mm512_store_epi64(data + i * 64 + sz/2 + 48, dv15);
			_mm512_store_epi64(data + i * 64 + sz/2 + 56, dv16);

		}
	}

	// two parallel half-size merges
	bitonic_merge(data, sz / 2, dir);
	bitonic_merge(data + sz / 2, sz / 2, dir);
}

//#define non_recursive
void bitonic_sort(uint64_t *data, uint32_t sz, int dir)
{
	if (sz == 64)
	{
		// base case: do the hardcoded 64-element sort
		bitonic_sort_dir_64(data, dir);
		return;
	}
	
#ifdef non_recursive
	// slower for large sorts: less cache efficient as 
	// written compared to the recursive version
	{
		int j;
		for (j = 0; j < sz / 64; j++) {
			// alternating up/down sorts so we can finish using merge only
			bitonic_sort_dir_64(data + j * 64, j & 1);
		}
		
		uint32_t bitonic_sort_size = 128;
		while (bitonic_sort_size < sz)
		{
			for (j = 0; j < sz / bitonic_sort_size; j++) {
				// up/down merges of the previous up/down sorts, output is up/down sorted
				bitonic_merge(&data[j * bitonic_sort_size], bitonic_sort_size, j & 1);
			}
			bitonic_sort_size *= 2;
		}
		
		// final merge in the specified direction
		bitonic_merge(data, sz, dir);
		return;
	}
#else
	
	// two parallel half-size bitonic sorts,
	// with opposite directions.
	bitonic_sort(data, sz / 2, 0);
	bitonic_sort(data + sz / 2, sz / 2, 1);
	
	// merge in the specified direction
	bitonic_merge(data, sz, dir);
#endif

	return;
}

void sort(uint64_t *data, uint32_t sz, int dir)
{
	// top level sort dealing with two things:
	// 1) the bitonic sort function requires the data array
	// to be aligned and
	// 2) the bitonic sort works on a power-of-2 sized array
	// here we make sure we feed the bitonic sort function
	// an array that satifies those requirements.
	int is_aligned = (((uint64_t)data & 0x3full) == 0);
	if (is_aligned && ((sz & (sz - 1)) == 0))
	{
		// meets both requirements as-is
		bitonic_sort(data, sz, dir);
		return;
	}
	
	// otherwise we need to copy to an aligned buffer and/or 
	// change the buffer size
	uint32_t new_sz = sz;
	if ((sz & (sz - 1)) > 0)
	{
		new_sz = my_clz32(sz);
		if (new_sz == 0)
		{
			printf("buffer too big, sz must be <= 2^31 in sort()\n");
			exit(0);
		}
		new_sz = 1 << (32 - new_sz + 1);
	}
		
	uint64_t *adata;
	
	if (!is_aligned)
	{
		adata = (uint64_t*)aligned_malloc(new_sz * sizeof(uint64_t), 64);
		memcpy(adata, data, sz * sizeof(uint64_t));
	}
	else
	{
		adata = data;
	}
	
	if ((new_sz - sz) > 0)
	{
		if (dir == 0)
			memset(adata + sz, 0xff, (new_sz - sz));
		else
			memset(adata + sz, 0, (new_sz - sz));
	}
	
	bitonic_sort(adata, new_sz, dir);
	
	if (!is_aligned)
	{
		memcpy(data, adata, sz * sizeof(uint64_t));
		aligned_free(adata);
	}
	
	return;
}

void bitonic_merge32(uint32_t *data, uint32_t sz, int dir)
{
	if (sz <= 128)
	{
		// base case: do the hardcoded 128-element sort
		bitonic_merge32_dir_128(data, dir);
		return;
	}
	
	// half-size cmp/swap
	int i;

	// we have sz/2 swaps to do at a stride of sz/2.
	// the number of swaps will be divisible by 128 because
	// sz is at least 256 (basecase is 128).
	// can batch up the swaps:
	__m512i t1;
	__m512i t2;
	__m512i t3;
	__m512i t4;
	__m512i t5;
	__m512i t6;
	__m512i t7;
	__m512i t8;
	__m512i dv1;
	__m512i dv2;
	__m512i dv3;
	__m512i dv4;
	__m512i dv5;
	__m512i dv6;
	__m512i dv7;
	__m512i dv8;
	__m512i dv9;
	__m512i dv10;
	__m512i dv11;
	__m512i dv12;
	__m512i dv13;
	__m512i dv14;
	__m512i dv15;
	__m512i dv16;

	__mmask16 m1;
	__mmask16 m2;
	__mmask16 m3;
	__mmask16 m4;
	__mmask16 m5;
	__mmask16 m6;
	__mmask16 m7;
	__mmask16 m8;
			
	if (dir == 1)
	{
		// 256-element merge passes at a stride of sz/2
		for (i = 0; i < sz / 256; i++)
		{
			// swap 128 elements at starting offset i * 128
			dv1 = _mm512_load_epi32(data + i * 128 +  0);
			dv2 = _mm512_load_epi32(data + i * 128 + 16);
			dv3 = _mm512_load_epi32(data + i * 128 + 32);
			dv4 = _mm512_load_epi32(data + i * 128 + 48);
			dv5 = _mm512_load_epi32(data + i * 128 + 64);
			dv6 = _mm512_load_epi32(data + i * 128 + 80);
			dv7 = _mm512_load_epi32(data + i * 128 + 96);
			dv8 = _mm512_load_epi32(data + i * 128 + 112);
			
			// with 128 elements at offset i * 128 + stride (sz/2)
			dv9 = _mm512_load_epi32(data  + i * 128 + sz/2 +  0);
			dv10 = _mm512_load_epi32(data + i * 128 + sz/2 + 16);
			dv11 = _mm512_load_epi32(data + i * 128 + sz/2 + 32);
			dv12 = _mm512_load_epi32(data + i * 128 + sz/2 + 48);
			dv13 = _mm512_load_epi32(data + i * 128 + sz/2 + 64);
			dv14 = _mm512_load_epi32(data + i * 128 + sz/2 + 80);
			dv15 = _mm512_load_epi32(data + i * 128 + sz/2 + 96);
			dv16 = _mm512_load_epi32(data + i * 128 + sz/2 + 112);

				m1 = _mm512_cmplt_epu32_mask(dv1, dv9);
				m2 = _mm512_cmplt_epu32_mask(dv2, dv10);
				m3 = _mm512_cmplt_epu32_mask(dv3, dv11);
				m4 = _mm512_cmplt_epu32_mask(dv4, dv12);
				m5 = _mm512_cmplt_epu32_mask(dv5, dv13);
				m6 = _mm512_cmplt_epu32_mask(dv6, dv14);
				m7 = _mm512_cmplt_epu32_mask(dv7, dv15);
				m8 = _mm512_cmplt_epu32_mask(dv8, dv16);


			t1 = _mm512_mask_blend_epi32(m1, dv1, dv9);
			t2 = _mm512_mask_blend_epi32(m2, dv2, dv10);
			t3 = _mm512_mask_blend_epi32(m3, dv3, dv11);
			t4 = _mm512_mask_blend_epi32(m4, dv4, dv12);
			t5 = _mm512_mask_blend_epi32(m5, dv5, dv13);
			t6 = _mm512_mask_blend_epi32(m6, dv6, dv14);
			t7 = _mm512_mask_blend_epi32(m7, dv7, dv15);
			t8 = _mm512_mask_blend_epi32(m8, dv8, dv16);
			
			dv9 = _mm512_mask_blend_epi32(m1, dv9, dv1);
			dv10 = _mm512_mask_blend_epi32(m2, dv10, dv2);
			dv11 = _mm512_mask_blend_epi32(m3, dv11, dv3);
			dv12 = _mm512_mask_blend_epi32(m4, dv12, dv4);
			dv13 = _mm512_mask_blend_epi32(m5, dv13, dv5);
			dv14 = _mm512_mask_blend_epi32(m6, dv14, dv6);
			dv15 = _mm512_mask_blend_epi32(m7, dv15, dv7);
			dv16 = _mm512_mask_blend_epi32(m8, dv16, dv8);

			dv1 = t1;
			dv2 = t2;
			dv3 = t3;
			dv4 = t4;
			dv5 = t5;
			dv6 = t6;
			dv7 = t7;
			dv8 = t8;
			
			_mm512_store_epi32(data + i * 128 +  0, dv1);
			_mm512_store_epi32(data + i * 128 + 16, dv2);
			_mm512_store_epi32(data + i * 128 + 32, dv3);
			_mm512_store_epi32(data + i * 128 + 48, dv4);
			_mm512_store_epi32(data + i * 128 + 64, dv5);
			_mm512_store_epi32(data + i * 128 + 80, dv6);
			_mm512_store_epi32(data + i * 128 + 96, dv7);
			_mm512_store_epi32(data + i * 128 + 112, dv8);
			
			_mm512_store_epi64(data + i * 128 + sz/2 +  0,  dv9);
			_mm512_store_epi64(data + i * 128 + sz/2 + 16,  dv10);
			_mm512_store_epi64(data + i * 128 + sz/2 + 32, dv11);
			_mm512_store_epi64(data + i * 128 + sz/2 + 48, dv12);
			_mm512_store_epi64(data + i * 128 + sz/2 + 64, dv13);
			_mm512_store_epi64(data + i * 128 + sz/2 + 80, dv14);
			_mm512_store_epi64(data + i * 128 + sz/2 + 96, dv15);
			_mm512_store_epi64(data + i * 128 + sz/2 + 112, dv16);

		}
	}
	else
	{
		// 256-element merge passes at a stride of sz/2
		for (i = 0; i < sz / 256; i++)
		{
			// swap 128 elements at starting offset i * 128
			dv1 = _mm512_load_epi32(data + i * 128 +  0);
			dv2 = _mm512_load_epi32(data + i * 128 + 16);
			dv3 = _mm512_load_epi32(data + i * 128 + 32);
			dv4 = _mm512_load_epi32(data + i * 128 + 48);
			dv5 = _mm512_load_epi32(data + i * 128 + 64);
			dv6 = _mm512_load_epi32(data + i * 128 + 80);
			dv7 = _mm512_load_epi32(data + i * 128 + 96);
			dv8 = _mm512_load_epi32(data + i * 128 + 112);
			
			// with 128 elements at offset i * 128 + stride (sz/2)
			dv9 = _mm512_load_epi32(data  + i * 128 + sz/2 +  0);
			dv10 = _mm512_load_epi32(data + i * 128 + sz/2 + 16);
			dv11 = _mm512_load_epi32(data + i * 128 + sz/2 + 32);
			dv12 = _mm512_load_epi32(data + i * 128 + sz/2 + 48);
			dv13 = _mm512_load_epi32(data + i * 128 + sz/2 + 64);
			dv14 = _mm512_load_epi32(data + i * 128 + sz/2 + 80);
			dv15 = _mm512_load_epi32(data + i * 128 + sz/2 + 96);
			dv16 = _mm512_load_epi32(data + i * 128 + sz/2 + 112);

				m1 = _mm512_cmpgt_epu32_mask(dv1, dv9);
				m2 = _mm512_cmpgt_epu32_mask(dv2, dv10);
				m3 = _mm512_cmpgt_epu32_mask(dv3, dv11);
				m4 = _mm512_cmpgt_epu32_mask(dv4, dv12);
				m5 = _mm512_cmpgt_epu32_mask(dv5, dv13);
				m6 = _mm512_cmpgt_epu32_mask(dv6, dv14);
				m7 = _mm512_cmpgt_epu32_mask(dv7, dv15);
				m8 = _mm512_cmpgt_epu32_mask(dv8, dv16);


			t1 = _mm512_mask_blend_epi32(m1, dv1, dv9);
			t2 = _mm512_mask_blend_epi32(m2, dv2, dv10);
			t3 = _mm512_mask_blend_epi32(m3, dv3, dv11);
			t4 = _mm512_mask_blend_epi32(m4, dv4, dv12);
			t5 = _mm512_mask_blend_epi32(m5, dv5, dv13);
			t6 = _mm512_mask_blend_epi32(m6, dv6, dv14);
			t7 = _mm512_mask_blend_epi32(m7, dv7, dv15);
			t8 = _mm512_mask_blend_epi32(m8, dv8, dv16);
			
			dv9 = _mm512_mask_blend_epi32(m1, dv9, dv1);
			dv10 = _mm512_mask_blend_epi32(m2, dv10, dv2);
			dv11 = _mm512_mask_blend_epi32(m3, dv11, dv3);
			dv12 = _mm512_mask_blend_epi32(m4, dv12, dv4);
			dv13 = _mm512_mask_blend_epi32(m5, dv13, dv5);
			dv14 = _mm512_mask_blend_epi32(m6, dv14, dv6);
			dv15 = _mm512_mask_blend_epi32(m7, dv15, dv7);
			dv16 = _mm512_mask_blend_epi32(m8, dv16, dv8);

			dv1 = t1;
			dv2 = t2;
			dv3 = t3;
			dv4 = t4;
			dv5 = t5;
			dv6 = t6;
			dv7 = t7;
			dv8 = t8;
			
			_mm512_store_epi32(data + i * 128 +  0, dv1);
			_mm512_store_epi32(data + i * 128 + 16, dv2);
			_mm512_store_epi32(data + i * 128 + 32, dv3);
			_mm512_store_epi32(data + i * 128 + 48, dv4);
			_mm512_store_epi32(data + i * 128 + 64, dv5);
			_mm512_store_epi32(data + i * 128 + 80, dv6);
			_mm512_store_epi32(data + i * 128 + 96, dv7);
			_mm512_store_epi32(data + i * 128 + 112, dv8);
			
			_mm512_store_epi64(data + i * 128 + sz/2 +  0,  dv9);
			_mm512_store_epi64(data + i * 128 + sz/2 + 16,  dv10);
			_mm512_store_epi64(data + i * 128 + sz/2 + 32, dv11);
			_mm512_store_epi64(data + i * 128 + sz/2 + 48, dv12);
			_mm512_store_epi64(data + i * 128 + sz/2 + 64, dv13);
			_mm512_store_epi64(data + i * 128 + sz/2 + 80, dv14);
			_mm512_store_epi64(data + i * 128 + sz/2 + 96, dv15);
			_mm512_store_epi64(data + i * 128 + sz/2 + 112, dv16);

		}
	}

	// two parallel half-size merges
	bitonic_merge32(data, sz / 2, dir);
	bitonic_merge32(data + sz / 2, sz / 2, dir);
}

void bitonic_sort32(uint32_t *data, uint32_t sz, int dir)
{
	if (sz == 256)
	{
		// base case: do the hardcoded 128-element sort
		bitonic_sort32_dir_256(data, dir);
		return;
	}

	// two half-size bitonic sorts,
	// with opposite directions.
	bitonic_sort32(data, sz / 2, 0);
	bitonic_sort32(data + sz / 2, sz / 2, 1);

	// merge in the specified direction
	bitonic_merge32(data, sz, dir);

	return;
}

void sort32(uint32_t *data, uint32_t sz, int dir)
{
	// top level sort dealing with two things:
	// 1) the bitonic sort function requires the data array
	// to be aligned and
	// 2) the bitonic sort works on a power-of-2 sized array
	// here we make sure we feed the bitonic sort function
	// an array that satifies those requirements.
	int is_aligned = (((uint64_t)data & 0x3full) == 0);
	if (is_aligned && ((sz & (sz - 1)) == 0))
	{
		// meets both requirements as-is
		bitonic_sort32(data, sz, dir);
		return;
	}
	
	// otherwise we need to copy to an aligned buffer and/or 
	// change the buffer size
	uint32_t new_sz = sz;
	if ((sz & (sz - 1)) > 0)
	{
		new_sz = my_clz32(sz);
		if (new_sz == 0)
		{
			printf("buffer too big, sz must be <= 2^31 in sort()\n");
			exit(0);
		}
		new_sz = 1 << (32 - new_sz + 1);
	}
		
	uint32_t *adata;
	
	if (!is_aligned)
	{
		adata = (uint32_t*)aligned_malloc(new_sz * sizeof(uint32_t), 64);
		memcpy(adata, data, sz * sizeof(uint32_t));
	}
	else
	{
		adata = data;
	}
	
	if ((new_sz - sz) > 0)
	{
		if (dir == 0)
			memset(adata + sz, 0xff, (new_sz - sz));
		else
			memset(adata + sz, 0, (new_sz - sz));
	}
	
	bitonic_sort32(adata, new_sz, dir);
	
	if (!is_aligned)
	{
		memcpy(data, adata, sz * sizeof(uint32_t));
		aligned_free(adata);
	}
	
	return;
}

void parsort(uint64_t *data, uint32_t sz, int dir, int threads)
{
	// REQUIRED: threads and sz both powers of 2 and
	// (threads * 64) divides sz
	uint32_t bitonic_sort_size = sz / threads;
	uint32_t j;
	
	omp_set_num_threads(threads);

#pragma omp parallel for
	for (j = 0; j < sz / bitonic_sort_size; j++) {
		// alternating up/down sorts so we can finish using merge only
		bitonic_sort(&data[j * bitonic_sort_size], bitonic_sort_size, j & 1);
	}
	
	if (threads == 1)
		return;
		
	bitonic_sort_size *= 2;
	while (bitonic_sort_size < sz)
	{
#pragma omp parallel for
		for (j = 0; j < sz / bitonic_sort_size; j++) {
			// up/down merges of the previous up/down sorts, output is up/down sorted
			bitonic_merge(&data[j * bitonic_sort_size], bitonic_sort_size, j & 1);
		}
		bitonic_sort_size *= 2;
	}
	
	// final merge in the specified direction
	bitonic_merge(data, sz, dir);
	
	return;
}

int main3(int argc, char ** argv)
{
	uint32 i, j, n;
	uint32 num_sort;
	uint32 key_bits;
	uint32 num_reps;
	uint32 num_threads;
	double seconds;
	double tseconds = 0.0;
	uint32 seed1 = 0x11111;
	uint32 seed2 = 0x22222;
	
	if ((argc < 5) || (argc > 6))
	{
		printf("usage: vecsort key_bits num_sort num_reps num_threads <seed>\n");
		exit(0);
	}

	key_bits = strtoul(argv[1], NULL, 10);
	num_sort = strtoul(argv[2], NULL, 10);
	num_reps = strtoul(argv[3], NULL, 10);
	num_threads = strtoul(argv[4], NULL, 10);
	
	if (argc == 6)
	{
		uint32 s = strtoul(argv[5], NULL, 10);
		seed1 = s & 0xffff;
		seed2 = s >> 16;
		printf("seed: %u\n", s);
	}
	
	uint64_t *loc_keys = (uint64 *)aligned_malloc(num_sort * sizeof(uint64), 64);
	
	for (n = 0; n < num_reps; n++) {
		
		for (i = 0; i < num_sort; i++) {
			uint64 key = (uint64)get_rand(&seed1, &seed2) << 32 |
					get_rand(&seed1, &seed2);
					
			loc_keys[i] = key >> (64 - key_bits);
		}
		
		seconds = get_cpu_time();
		
		uint32 num_collisions = 0;
		
		parsort(loc_keys, num_sort, 0, num_threads);
		
		tseconds += (get_cpu_time() - seconds);
		
		for (j = 1; j < num_sort; j++) {
			if (loc_keys[j] < loc_keys[j-1]) {
				printf("sort error\n");
				goto done;
			}
			
			if (loc_keys[j] == loc_keys[j-1]) {
				num_collisions++;
			}
		}
		
		if (num_reps <= 10)
			printf("found %u total collisions in %u elements\n", 
				num_collisions, num_sort);
	}
	
done:
	printf("sort %u x %u-bit keys in average of %lf seconds\n", 
		num_sort, key_bits, tseconds / (double)num_reps);
	
	aligned_free(loc_keys);
	return 0;
}

int main(int argc, char ** argv)
{
	uint32 i, j, n;
	uint32 num_sort;
	uint32 num_reps;
	uint32 key_bits;
	uint32 sort_sz;
	double seconds;
	double tseconds = 0.0;
	uint32 seed1 = 0x11111;
	uint32 seed2 = 0x22222;
	
	if ((argc < 5) || (argc > 6))
	{
		printf("usage: vecsort sort_sz_bits key_bits num_sort num_reps <seed>\n");
		exit(0);
	}
	
	sort_sz = strtoul(argv[1], NULL, 10);
	key_bits = strtoul(argv[2], NULL, 10);
	num_sort = strtoul(argv[3], NULL, 10);
	num_reps = strtoul(argv[4], NULL, 10);
	
	if (argc == 7)
	{
		uint32 s = strtoul(argv[5], NULL, 10);
		seed1 = s & 0xffff;
		seed2 = s >> 16;
		printf("seed: %u\n", s);
	}

	uint32 bitonic_sort_size = (1 << sort_sz);

	uint32 num_lists = num_sort / bitonic_sort_size;
	
	if (num_lists * bitonic_sort_size < num_sort)
	{
		num_lists++;
		num_sort = num_lists * bitonic_sort_size;
	}

	uint64_t *loc_keys = (uint64 *)aligned_malloc(num_sort * sizeof(uint64), 64);
	
	for (n = 0; n < num_reps; n++) {
		
		for (i = 0; i < num_sort; i++) {
			uint64 key = (uint64)get_rand(&seed1, &seed2) << 32 |
					get_rand(&seed1, &seed2);
					
			loc_keys[i] = key >> (64 - key_bits);
		}
		
		seconds = get_cpu_time();
		
		uint32 num_collisions = 0;
		
		for (j = 0; j < num_sort; j += bitonic_sort_size) {
			sort(loc_keys + j, bitonic_sort_size, 0);
			//bitonic_sort_dir_64(loc_keys + j, 0);
		}
		
		tseconds += (get_cpu_time() - seconds);
		
		for (i = 0; i < num_lists; i++) {
			for (j = 1; j < bitonic_sort_size; j++) {
				if (loc_keys[i * bitonic_sort_size + j] < loc_keys[i * bitonic_sort_size + j-1]) {
					printf("sort error\n");
					goto done;
				}
				
				if (loc_keys[i * bitonic_sort_size + j] == loc_keys[i * bitonic_sort_size + j-1]) {
					num_collisions++;
				}
			}
		}
		
		printf("found %u total collisions in %u lists of %u elements\n", 
			num_collisions, num_lists, bitonic_sort_size);
	}
	
	printf("sort %u x %u-bit keys in average of %lf seconds\n", 
		num_sort, key_bits, tseconds / (double)num_reps);
	
	tseconds = 0.0;
	for (n = 0; n < num_reps; n++) {
		
		for (i = 0; i < num_sort; i++) {
			uint64 key = (uint64)get_rand(&seed1, &seed2) << 32 |
					get_rand(&seed1, &seed2);
					
			loc_keys[i] = key >> (64 - key_bits);
		}
		
		seconds = get_cpu_time();
		
		uint32 num_collisions = 0;
		
		for (j = 0; j < num_sort; j += bitonic_sort_size) {
			qsort(loc_keys + j, bitonic_sort_size, sizeof(uint64), &qcomp_uint64);
		}
		
		tseconds += (get_cpu_time() - seconds);
		
		for (i = 0; i < num_lists; i++) {
			for (j = 1; j < bitonic_sort_size; j++) {
				if (loc_keys[i * bitonic_sort_size + j] < loc_keys[i * bitonic_sort_size + j-1]) {
					printf("sort error at position %d in list\n", j);
					
					int k;
					for (k = 0; k < bitonic_sort_size; k++)
					{
						if (k % 16 == 0) printf("\n");
						printf("%16lu ", loc_keys[i * bitonic_sort_size + k]);
					}
					printf("\n");
	
					goto done;
				}
				
				if (loc_keys[i * bitonic_sort_size + j] == loc_keys[i * bitonic_sort_size + j-1]) {
					num_collisions++;
				}
			}
		}
		
		printf("found %u total collisions in %u lists of %u elements\n", 
			num_collisions, num_lists, bitonic_sort_size);
	}
	
done:
	printf("sort %u x %u-bit keys in average of %lf seconds\n", 
		num_sort, key_bits, tseconds / (double)num_reps);
	
	aligned_free(loc_keys);
	return 0;
}

int main2(int argc, char ** argv)
{
	uint32 i, j, n;
	uint32 num_sort;
	uint32 num_reps;
	uint32 key_bits;
	uint32 sort_sz;
	double seconds;
	double tseconds = 0.0;
	uint32 seed1 = 0x11111;
	uint32 seed2 = 0x22222;
	
	swap8bit_idx = _mm512_set1_epi64(0xefcdba8967452301);
	swap16bit_idx = _mm512_set1_epi64(0xdcfe98ba54761032);
	
	if ((argc < 5) || (argc > 6))
	{
		printf("usage: vecsort sort_sz_bits key_bits num_sort num_reps <seed>\n");
		exit(0);
	}
	
	sort_sz = strtoul(argv[1], NULL, 10);
	key_bits = strtoul(argv[2], NULL, 10);
	num_sort = strtoul(argv[3], NULL, 10);
	num_reps = strtoul(argv[4], NULL, 10);
	
	if (argc == 7)
	{
		uint32 s = strtoul(argv[5], NULL, 10);
		seed1 = s & 0xffff;
		seed2 = s >> 16;
		printf("seed: %u\n", s);
	}

	uint32 bitonic_sort_size = (1 << sort_sz);

	uint32 num_lists = num_sort / bitonic_sort_size;
	
	if (num_lists * bitonic_sort_size < num_sort)
	{
		num_lists++;
		num_sort = num_lists * bitonic_sort_size;
	}

	uint32_t *loc_keys = (uint32 *)aligned_malloc(num_sort * sizeof(uint32), 64);
	
	for (n = 0; n < num_reps; n++) {
		
		for (i = 0; i < num_sort; i++) {
			uint32 key = get_rand(&seed1, &seed2);
			loc_keys[i] = key >> (32 - key_bits);
		}
		
		seconds = get_cpu_time();
		
		uint32 num_collisions = 0;
		
		for (j = 0; j < num_sort; j += bitonic_sort_size) {
			//bitonic_sort32_dir_128(loc_keys + j, 0);
			//bitonic_sort32_dir_256(loc_keys + j, 0);
			sort32(loc_keys + j, bitonic_sort_size, 0);
		}
		
		tseconds += (get_cpu_time() - seconds);
		
		for (i = 0; i < num_lists; i++) {
			for (j = 1; j < bitonic_sort_size; j++) {
				if (loc_keys[i * bitonic_sort_size + j] < loc_keys[i * bitonic_sort_size + j-1]) {
					printf("sort error at position %d in list\n", j);
					
					int k;
					for (k = 0; k < bitonic_sort_size; k++)
					{
						if (k % 16 == 0) printf("\n");
						printf("%08u ", loc_keys[i * bitonic_sort_size + k]);
					}
					printf("\n");
				
					goto done;
				}
				
				if (loc_keys[i * bitonic_sort_size + j] == loc_keys[i * bitonic_sort_size + j-1]) {
					num_collisions++;
				}
			}
		}
		
		printf("found %u total collisions in %u lists of %u elements\n", 
			num_collisions, num_lists, bitonic_sort_size);
	}
	
	printf("sort %u x %u-bit keys in average of %lf seconds\n", 
		num_sort, key_bits, tseconds / (double)num_reps);
	
	tseconds = 0.0;
	for (n = 0; n < num_reps; n++) {
		
		for (i = 0; i < num_sort; i++) {
			uint32 key = get_rand(&seed1, &seed2);
			loc_keys[i] = key >> (32 - key_bits);
		}
		
		seconds = get_cpu_time();
		
		uint32 num_collisions = 0;
		
		for (j = 0; j < num_sort; j += bitonic_sort_size) {
			qsort(loc_keys + j, bitonic_sort_size, sizeof(uint32), &qcomp_uint32);
		}
		
		tseconds += (get_cpu_time() - seconds);
		
		for (i = 0; i < num_lists; i++) {
			for (j = 1; j < bitonic_sort_size; j++) {
				if (loc_keys[i * bitonic_sort_size + j] < loc_keys[i * bitonic_sort_size + j-1]) {
					printf("sort error at position %d in list\n", j);
					
					int k;
					for (k = 0; k < bitonic_sort_size; k++)
					{
						if (k % 16 == 0) printf("\n");
						printf("%08u ", loc_keys[i * bitonic_sort_size + k]);
					}
					printf("\n");
	
					goto done;
				}
				
				if (loc_keys[i * bitonic_sort_size + j] == loc_keys[i * bitonic_sort_size + j-1]) {
					num_collisions++;
				}
			}
		}
		
		printf("found %u total collisions in %u lists of %u elements\n", 
			num_collisions, num_lists, bitonic_sort_size);
	}
	
done:
	printf("sort %u x %u-bit keys in average of %lf seconds\n", 
		num_sort, key_bits, tseconds / (double)num_reps);
	
	aligned_free(loc_keys);
	return 0;
}




	