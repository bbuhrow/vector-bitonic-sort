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

void bitonic_merge_dir_64(uint64_t* data, int dir)
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
		dv1_swap = _mm512_shuffle_i64x2(dv1, dv1, 0x4E);
		dv2_swap = _mm512_shuffle_i64x2(dv2, dv2, 0x4E);
		dv3_swap = _mm512_shuffle_i64x2(dv3, dv3, 0x4E);
		dv4_swap = _mm512_shuffle_i64x2(dv4, dv4, 0x4E);
		dv5_swap = _mm512_shuffle_i64x2(dv5, dv5, 0x4E);
		dv6_swap = _mm512_shuffle_i64x2(dv6, dv6, 0x4E);
		dv7_swap = _mm512_shuffle_i64x2(dv7, dv7, 0x4E);
		dv8_swap = _mm512_shuffle_i64x2(dv8, dv8, 0x4E);
		
		m1 = _mm512_cmplt_epu64_mask(dv1, dv1_swap);
		m2 = _mm512_cmplt_epu64_mask(dv2, dv2_swap);
		m3 = _mm512_cmplt_epu64_mask(dv3, dv3_swap);
		m4 = _mm512_cmplt_epu64_mask(dv4, dv4_swap);
		m5 = _mm512_cmplt_epu64_mask(dv5, dv5_swap);
		m6 = _mm512_cmplt_epu64_mask(dv6, dv6_swap);
		m7 = _mm512_cmplt_epu64_mask(dv7, dv7_swap);
		m8 = _mm512_cmplt_epu64_mask(dv8, dv8_swap);

		// '0F' for the swap between dist-4 lanes, non alternating
		dv1 = _mm512_mask_blend_epi64(m1 ^ 0x0F, dv1_swap, dv1);
		dv2 = _mm512_mask_blend_epi64(m2 ^ 0x0F, dv2_swap, dv2);
		dv3 = _mm512_mask_blend_epi64(m3 ^ 0x0F, dv3_swap, dv3);
		dv4 = _mm512_mask_blend_epi64(m4 ^ 0x0F, dv4_swap, dv4);
		dv5 = _mm512_mask_blend_epi64(m5 ^ 0x0F, dv5_swap, dv5);
		dv6 = _mm512_mask_blend_epi64(m6 ^ 0x0F, dv6_swap, dv6);
		dv7 = _mm512_mask_blend_epi64(m7 ^ 0x0F, dv7_swap, dv7);
		dv8 = _mm512_mask_blend_epi64(m8 ^ 0x0F, dv8_swap, dv8);

		// distance 2 swaps
		dv1_swap = _mm512_permutex_epi64(dv1, 0x4E);
		dv2_swap = _mm512_permutex_epi64(dv2, 0x4E);
		dv3_swap = _mm512_permutex_epi64(dv3, 0x4E);
		dv4_swap = _mm512_permutex_epi64(dv4, 0x4E);
		dv5_swap = _mm512_permutex_epi64(dv5, 0x4E);
		dv6_swap = _mm512_permutex_epi64(dv6, 0x4E);
		dv7_swap = _mm512_permutex_epi64(dv7, 0x4E);
		dv8_swap = _mm512_permutex_epi64(dv8, 0x4E);
		
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
		dv1_swap = _mm512_shuffle_epi32(dv1, 0x4E);
		dv2_swap = _mm512_shuffle_epi32(dv2, 0x4E);
		dv3_swap = _mm512_shuffle_epi32(dv3, 0x4E);
		dv4_swap = _mm512_shuffle_epi32(dv4, 0x4E);
		dv5_swap = _mm512_shuffle_epi32(dv5, 0x4E);
		dv6_swap = _mm512_shuffle_epi32(dv6, 0x4E);
		dv7_swap = _mm512_shuffle_epi32(dv7, 0x4E);
		dv8_swap = _mm512_shuffle_epi32(dv8, 0x4E);
		
		m1 = _mm512_cmplt_epu64_mask(dv1_swap, dv1);
		m2 = _mm512_cmplt_epu64_mask(dv2_swap, dv2);
		m3 = _mm512_cmplt_epu64_mask(dv3_swap, dv3);
		m4 = _mm512_cmplt_epu64_mask(dv4_swap, dv4);
		m5 = _mm512_cmplt_epu64_mask(dv5_swap, dv5);
		m6 = _mm512_cmplt_epu64_mask(dv6_swap, dv6);
		m7 = _mm512_cmplt_epu64_mask(dv7_swap, dv7);
		m8 = _mm512_cmplt_epu64_mask(dv8_swap, dv8);

		// 'AA' for the swap between adjacent lanes
		dv1 = _mm512_mask_blend_epi64(m1 ^ 0xaa, dv1_swap, dv1);
		dv2 = _mm512_mask_blend_epi64(m2 ^ 0xaa, dv2_swap, dv2);
		dv3 = _mm512_mask_blend_epi64(m3 ^ 0xaa, dv3_swap, dv3);
		dv4 = _mm512_mask_blend_epi64(m4 ^ 0xaa, dv4_swap, dv4);
		dv5 = _mm512_mask_blend_epi64(m5 ^ 0xaa, dv5_swap, dv5);
		dv6 = _mm512_mask_blend_epi64(m6 ^ 0xaa, dv6_swap, dv6);
		dv7 = _mm512_mask_blend_epi64(m7 ^ 0xaa, dv7_swap, dv7);
		dv8 = _mm512_mask_blend_epi64(m8 ^ 0xaa, dv8_swap, dv8);

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
		dv1_swap = _mm512_shuffle_i64x2(dv1, dv1, 0x4E);
		dv2_swap = _mm512_shuffle_i64x2(dv2, dv2, 0x4E);
		dv3_swap = _mm512_shuffle_i64x2(dv3, dv3, 0x4E);
		dv4_swap = _mm512_shuffle_i64x2(dv4, dv4, 0x4E);
		dv5_swap = _mm512_shuffle_i64x2(dv5, dv5, 0x4E);
		dv6_swap = _mm512_shuffle_i64x2(dv6, dv6, 0x4E);
		dv7_swap = _mm512_shuffle_i64x2(dv7, dv7, 0x4E);
		dv8_swap = _mm512_shuffle_i64x2(dv8, dv8, 0x4E);
		m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
		m2 = _mm512_cmpgt_epu64_mask(dv2, dv2_swap);
		m3 = _mm512_cmpgt_epu64_mask(dv3, dv3_swap);
		m4 = _mm512_cmpgt_epu64_mask(dv4, dv4_swap);
		m5 = _mm512_cmpgt_epu64_mask(dv5, dv5_swap);
		m6 = _mm512_cmpgt_epu64_mask(dv6, dv6_swap);
		m7 = _mm512_cmpgt_epu64_mask(dv7, dv7_swap);
		m8 = _mm512_cmpgt_epu64_mask(dv8, dv8_swap);

		// '0F' for the swap between dist-4 lanes, non alternating
		dv1 = _mm512_mask_blend_epi64(m1 ^ 0x0F, dv1_swap, dv1);
		dv2 = _mm512_mask_blend_epi64(m2 ^ 0x0F, dv2_swap, dv2);
		dv3 = _mm512_mask_blend_epi64(m3 ^ 0x0F, dv3_swap, dv3);
		dv4 = _mm512_mask_blend_epi64(m4 ^ 0x0F, dv4_swap, dv4);
		dv5 = _mm512_mask_blend_epi64(m5 ^ 0x0F, dv5_swap, dv5);
		dv6 = _mm512_mask_blend_epi64(m6 ^ 0x0F, dv6_swap, dv6);
		dv7 = _mm512_mask_blend_epi64(m7 ^ 0x0F, dv7_swap, dv7);
		dv8 = _mm512_mask_blend_epi64(m8 ^ 0x0F, dv8_swap, dv8);

		// distance 2 swaps
		dv1_swap = _mm512_permutex_epi64(dv1, 0x4E);
		dv2_swap = _mm512_permutex_epi64(dv2, 0x4E);
		dv3_swap = _mm512_permutex_epi64(dv3, 0x4E);
		dv4_swap = _mm512_permutex_epi64(dv4, 0x4E);
		dv5_swap = _mm512_permutex_epi64(dv5, 0x4E);
		dv6_swap = _mm512_permutex_epi64(dv6, 0x4E);
		dv7_swap = _mm512_permutex_epi64(dv7, 0x4E);
		dv8_swap = _mm512_permutex_epi64(dv8, 0x4E);
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
		dv1_swap = _mm512_shuffle_epi32(dv1, 0x4E);
		dv2_swap = _mm512_shuffle_epi32(dv2, 0x4E);
		dv3_swap = _mm512_shuffle_epi32(dv3, 0x4E);
		dv4_swap = _mm512_shuffle_epi32(dv4, 0x4E);
		dv5_swap = _mm512_shuffle_epi32(dv5, 0x4E);
		dv6_swap = _mm512_shuffle_epi32(dv6, 0x4E);
		dv7_swap = _mm512_shuffle_epi32(dv7, 0x4E);
		dv8_swap = _mm512_shuffle_epi32(dv8, 0x4E);
		m1 = _mm512_cmpgt_epu64_mask(dv1_swap, dv1);
		m2 = _mm512_cmpgt_epu64_mask(dv2_swap, dv2);
		m3 = _mm512_cmpgt_epu64_mask(dv3_swap, dv3);
		m4 = _mm512_cmpgt_epu64_mask(dv4_swap, dv4);
		m5 = _mm512_cmpgt_epu64_mask(dv5_swap, dv5);
		m6 = _mm512_cmpgt_epu64_mask(dv6_swap, dv6);
		m7 = _mm512_cmpgt_epu64_mask(dv7_swap, dv7);
		m8 = _mm512_cmpgt_epu64_mask(dv8_swap, dv8);

		// 'AA' for the swap between adjacent lanes
		dv1 = _mm512_mask_blend_epi64(m1 ^ 0xaa, dv1_swap, dv1);
		dv2 = _mm512_mask_blend_epi64(m2 ^ 0xaa, dv2_swap, dv2);
		dv3 = _mm512_mask_blend_epi64(m3 ^ 0xaa, dv3_swap, dv3);
		dv4 = _mm512_mask_blend_epi64(m4 ^ 0xaa, dv4_swap, dv4);
		dv5 = _mm512_mask_blend_epi64(m5 ^ 0xaa, dv5_swap, dv5);
		dv6 = _mm512_mask_blend_epi64(m6 ^ 0xaa, dv6_swap, dv6);
		dv7 = _mm512_mask_blend_epi64(m7 ^ 0xaa, dv7_swap, dv7);
		dv8 = _mm512_mask_blend_epi64(m8 ^ 0xaa, dv8_swap, dv8);
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


	
	// adjacent swaps, alternating compares
	dv1_swap = _mm512_shuffle_epi32(dv1, 0x4E);
	dv2_swap = _mm512_shuffle_epi32(dv2, 0x4E);
	dv3_swap = _mm512_shuffle_epi32(dv3, 0x4E);
	dv4_swap = _mm512_shuffle_epi32(dv4, 0x4E);
	dv5_swap = _mm512_shuffle_epi32(dv5, 0x4E);
	dv6_swap = _mm512_shuffle_epi32(dv6, 0x4E);
	dv7_swap = _mm512_shuffle_epi32(dv7, 0x4E);
	dv8_swap = _mm512_shuffle_epi32(dv8, 0x4E);
	
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
	

	// distance 2 swaps
	dv1_swap = _mm512_permutex_epi64(dv1, 0x4E);
	dv2_swap = _mm512_permutex_epi64(dv2, 0x4E);
	dv3_swap = _mm512_permutex_epi64(dv3, 0x4E);
	dv4_swap = _mm512_permutex_epi64(dv4, 0x4E);
	dv5_swap = _mm512_permutex_epi64(dv5, 0x4E);
	dv6_swap = _mm512_permutex_epi64(dv6, 0x4E);
	dv7_swap = _mm512_permutex_epi64(dv7, 0x4E);
	dv8_swap = _mm512_permutex_epi64(dv8, 0x4E);
	
	m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
	m2 = _mm512_cmpgt_epu64_mask(dv2, dv2_swap);
	m3 = _mm512_cmpgt_epu64_mask(dv3, dv3_swap);
	m4 = _mm512_cmpgt_epu64_mask(dv4, dv4_swap);
	m5 = _mm512_cmpgt_epu64_mask(dv5, dv5_swap);
	m6 = _mm512_cmpgt_epu64_mask(dv6, dv6_swap);
	m7 = _mm512_cmpgt_epu64_mask(dv7, dv7_swap);
	m8 = _mm512_cmpgt_epu64_mask(dv8, dv8_swap);

	// 'CC' for the swap between dist-2 lanes ^ '0F' to alternate gt vs. lt --> 0xC3
	dv1 = _mm512_mask_blend_epi64(m1 ^ 0xC3, dv1, dv1_swap);
	dv2 = _mm512_mask_blend_epi64(m2 ^ 0xC3, dv2, dv2_swap);
	dv3 = _mm512_mask_blend_epi64(m3 ^ 0xC3, dv3, dv3_swap);
	dv4 = _mm512_mask_blend_epi64(m4 ^ 0xC3, dv4, dv4_swap);
	dv5 = _mm512_mask_blend_epi64(m5 ^ 0xC3, dv5, dv5_swap);
	dv6 = _mm512_mask_blend_epi64(m6 ^ 0xC3, dv6, dv6_swap);
	dv7 = _mm512_mask_blend_epi64(m7 ^ 0xC3, dv7, dv7_swap);
	dv8 = _mm512_mask_blend_epi64(m8 ^ 0xC3, dv8, dv8_swap);

	// adjacent swaps
	dv1_swap = _mm512_shuffle_epi32(dv1, 0x4E);
	dv2_swap = _mm512_shuffle_epi32(dv2, 0x4E);
	dv3_swap = _mm512_shuffle_epi32(dv3, 0x4E);
	dv4_swap = _mm512_shuffle_epi32(dv4, 0x4E);
	dv5_swap = _mm512_shuffle_epi32(dv5, 0x4E);
	dv6_swap = _mm512_shuffle_epi32(dv6, 0x4E);
	dv7_swap = _mm512_shuffle_epi32(dv7, 0x4E);
	dv8_swap = _mm512_shuffle_epi32(dv8, 0x4E);
	
	m1 = _mm512_cmpgt_epu64_mask(dv1_swap, dv1);
	m2 = _mm512_cmpgt_epu64_mask(dv2_swap, dv2);
	m3 = _mm512_cmpgt_epu64_mask(dv3_swap, dv3);
	m4 = _mm512_cmpgt_epu64_mask(dv4_swap, dv4);
	m5 = _mm512_cmpgt_epu64_mask(dv5_swap, dv5);
	m6 = _mm512_cmpgt_epu64_mask(dv6_swap, dv6);
	m7 = _mm512_cmpgt_epu64_mask(dv7_swap, dv7);
	m8 = _mm512_cmpgt_epu64_mask(dv8_swap, dv8);

	// 'AA' for the swap between adjacent lanes ^ '0F' to alternate gt vs. lt --> 0xA5
	dv1 = _mm512_mask_blend_epi64(m1 ^ 0xA5, dv1_swap, dv1);
	dv2 = _mm512_mask_blend_epi64(m2 ^ 0xA5, dv2_swap, dv2);
	dv3 = _mm512_mask_blend_epi64(m3 ^ 0xA5, dv3_swap, dv3);
	dv4 = _mm512_mask_blend_epi64(m4 ^ 0xA5, dv4_swap, dv4);
	dv5 = _mm512_mask_blend_epi64(m5 ^ 0xA5, dv5_swap, dv5);
	dv6 = _mm512_mask_blend_epi64(m6 ^ 0xA5, dv6_swap, dv6);
	dv7 = _mm512_mask_blend_epi64(m7 ^ 0xA5, dv7_swap, dv7);
	dv8 = _mm512_mask_blend_epi64(m8 ^ 0xA5, dv8_swap, dv8);


	// phase 2

	// distance 4 swaps
	dv1_swap = _mm512_shuffle_i64x2(dv1, dv1, 0x4E);
	dv2_swap = _mm512_shuffle_i64x2(dv2, dv2, 0x4E);
	dv3_swap = _mm512_shuffle_i64x2(dv3, dv3, 0x4E);
	dv4_swap = _mm512_shuffle_i64x2(dv4, dv4, 0x4E);
	dv5_swap = _mm512_shuffle_i64x2(dv5, dv5, 0x4E);
	dv6_swap = _mm512_shuffle_i64x2(dv6, dv6, 0x4E);
	dv7_swap = _mm512_shuffle_i64x2(dv7, dv7, 0x4E);
	dv8_swap = _mm512_shuffle_i64x2(dv8, dv8, 0x4E);
	
	m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
	m2 = _mm512_cmplt_epu64_mask(dv2, dv2_swap);
	m3 = _mm512_cmpgt_epu64_mask(dv3, dv3_swap);
	m4 = _mm512_cmplt_epu64_mask(dv4, dv4_swap);
	m5 = _mm512_cmpgt_epu64_mask(dv5, dv5_swap);
	m6 = _mm512_cmplt_epu64_mask(dv6, dv6_swap);
	m7 = _mm512_cmpgt_epu64_mask(dv7, dv7_swap);
	m8 = _mm512_cmplt_epu64_mask(dv8, dv8_swap);

	// '0F' for the swap between dist-4 lanes, non alternating
	dv1 = _mm512_mask_blend_epi64(m1 ^ 0x0F, dv1, dv1_swap);
	dv2 = _mm512_mask_blend_epi64(m2 ^ 0x0F, dv2, dv2_swap);
	dv3 = _mm512_mask_blend_epi64(m3 ^ 0x0F, dv3, dv3_swap);
	dv4 = _mm512_mask_blend_epi64(m4 ^ 0x0F, dv4, dv4_swap);
	dv5 = _mm512_mask_blend_epi64(m5 ^ 0x0F, dv5, dv5_swap);
	dv6 = _mm512_mask_blend_epi64(m6 ^ 0x0F, dv6, dv6_swap);
	dv7 = _mm512_mask_blend_epi64(m7 ^ 0x0F, dv7, dv7_swap);
	dv8 = _mm512_mask_blend_epi64(m8 ^ 0x0F, dv8, dv8_swap);

	// distance 2 swaps
	dv1_swap = _mm512_permutex_epi64(dv1, 0x4E);
	dv2_swap = _mm512_permutex_epi64(dv2, 0x4E);
	dv3_swap = _mm512_permutex_epi64(dv3, 0x4E);
	dv4_swap = _mm512_permutex_epi64(dv4, 0x4E);
	dv5_swap = _mm512_permutex_epi64(dv5, 0x4E);
	dv6_swap = _mm512_permutex_epi64(dv6, 0x4E);
	dv7_swap = _mm512_permutex_epi64(dv7, 0x4E);
	dv8_swap = _mm512_permutex_epi64(dv8, 0x4E);
	
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
	dv1_swap = _mm512_shuffle_epi32(dv1, 0x4E);
	dv2_swap = _mm512_shuffle_epi32(dv2, 0x4E);
	dv3_swap = _mm512_shuffle_epi32(dv3, 0x4E);
	dv4_swap = _mm512_shuffle_epi32(dv4, 0x4E);
	dv5_swap = _mm512_shuffle_epi32(dv5, 0x4E);
	dv6_swap = _mm512_shuffle_epi32(dv6, 0x4E);
	dv7_swap = _mm512_shuffle_epi32(dv7, 0x4E);
	dv8_swap = _mm512_shuffle_epi32(dv8, 0x4E);
	
	m1 = _mm512_cmpgt_epu64_mask(dv1_swap, dv1);
	m2 = _mm512_cmplt_epu64_mask(dv2_swap, dv2);
	m3 = _mm512_cmpgt_epu64_mask(dv3_swap, dv3);
	m4 = _mm512_cmplt_epu64_mask(dv4_swap, dv4);
	m5 = _mm512_cmpgt_epu64_mask(dv5_swap, dv5);
	m6 = _mm512_cmplt_epu64_mask(dv6_swap, dv6);
	m7 = _mm512_cmpgt_epu64_mask(dv7_swap, dv7);
	m8 = _mm512_cmplt_epu64_mask(dv8_swap, dv8);

	// 'AA' for the swap between adjacent lanes, non alternating
	dv1 = _mm512_mask_blend_epi64(m1 ^ 0xAA, dv1_swap, dv1);
	dv2 = _mm512_mask_blend_epi64(m2 ^ 0xAA, dv2_swap, dv2);
	dv3 = _mm512_mask_blend_epi64(m3 ^ 0xAA, dv3_swap, dv3);
	dv4 = _mm512_mask_blend_epi64(m4 ^ 0xAA, dv4_swap, dv4);
	dv5 = _mm512_mask_blend_epi64(m5 ^ 0xAA, dv5_swap, dv5);
	dv6 = _mm512_mask_blend_epi64(m6 ^ 0xAA, dv6_swap, dv6);
	dv7 = _mm512_mask_blend_epi64(m7 ^ 0xAA, dv7_swap, dv7);
	dv8 = _mm512_mask_blend_epi64(m8 ^ 0xAA, dv8_swap, dv8);


	// phase 3

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
	dv1_swap = _mm512_shuffle_i64x2(dv1, dv1, 0x4E);
	dv2_swap = _mm512_shuffle_i64x2(dv2, dv2, 0x4E);
	dv3_swap = _mm512_shuffle_i64x2(dv3, dv3, 0x4E);
	dv4_swap = _mm512_shuffle_i64x2(dv4, dv4, 0x4E);
	dv5_swap = _mm512_shuffle_i64x2(dv5, dv5, 0x4E);
	dv6_swap = _mm512_shuffle_i64x2(dv6, dv6, 0x4E);
	dv7_swap = _mm512_shuffle_i64x2(dv7, dv7, 0x4E);
	dv8_swap = _mm512_shuffle_i64x2(dv8, dv8, 0x4E);
	
	m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
	m2 = _mm512_cmpgt_epu64_mask(dv2, dv2_swap);
	m3 = _mm512_cmplt_epu64_mask(dv3, dv3_swap);
	m4 = _mm512_cmplt_epu64_mask(dv4, dv4_swap);
	m5 = _mm512_cmpgt_epu64_mask(dv5, dv5_swap);
	m6 = _mm512_cmpgt_epu64_mask(dv6, dv6_swap);
	m7 = _mm512_cmplt_epu64_mask(dv7, dv7_swap);
	m8 = _mm512_cmplt_epu64_mask(dv8, dv8_swap);

	// '0F' for the swap between dist-4 lanes, non alternating
	dv1 = _mm512_mask_blend_epi64(m1 ^ 0x0F, dv1_swap, dv1);
	dv2 = _mm512_mask_blend_epi64(m2 ^ 0x0F, dv2_swap, dv2);
	dv3 = _mm512_mask_blend_epi64(m3 ^ 0x0F, dv3_swap, dv3);
	dv4 = _mm512_mask_blend_epi64(m4 ^ 0x0F, dv4_swap, dv4);
	dv5 = _mm512_mask_blend_epi64(m5 ^ 0x0F, dv5_swap, dv5);
	dv6 = _mm512_mask_blend_epi64(m6 ^ 0x0F, dv6_swap, dv6);
	dv7 = _mm512_mask_blend_epi64(m7 ^ 0x0F, dv7_swap, dv7);
	dv8 = _mm512_mask_blend_epi64(m8 ^ 0x0F, dv8_swap, dv8);

	// distance 2 swaps
	dv1_swap = _mm512_permutex_epi64(dv1, 0x4E);
	dv2_swap = _mm512_permutex_epi64(dv2, 0x4E);
	dv3_swap = _mm512_permutex_epi64(dv3, 0x4E);
	dv4_swap = _mm512_permutex_epi64(dv4, 0x4E);
	dv5_swap = _mm512_permutex_epi64(dv5, 0x4E);
	dv6_swap = _mm512_permutex_epi64(dv6, 0x4E);
	dv7_swap = _mm512_permutex_epi64(dv7, 0x4E);
	dv8_swap = _mm512_permutex_epi64(dv8, 0x4E);
	
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
	dv1_swap = _mm512_shuffle_epi32(dv1, 0x4E);
	dv2_swap = _mm512_shuffle_epi32(dv2, 0x4E);
	dv3_swap = _mm512_shuffle_epi32(dv3, 0x4E);
	dv4_swap = _mm512_shuffle_epi32(dv4, 0x4E);
	dv5_swap = _mm512_shuffle_epi32(dv5, 0x4E);
	dv6_swap = _mm512_shuffle_epi32(dv6, 0x4E);
	dv7_swap = _mm512_shuffle_epi32(dv7, 0x4E);
	dv8_swap = _mm512_shuffle_epi32(dv8, 0x4E);
	
	m1 = _mm512_cmpgt_epu64_mask(dv1_swap, dv1);
	m2 = _mm512_cmpgt_epu64_mask(dv2_swap, dv2);
	m3 = _mm512_cmplt_epu64_mask(dv3_swap, dv3);
	m4 = _mm512_cmplt_epu64_mask(dv4_swap, dv4);
	m5 = _mm512_cmpgt_epu64_mask(dv5_swap, dv5);
	m6 = _mm512_cmpgt_epu64_mask(dv6_swap, dv6);
	m7 = _mm512_cmplt_epu64_mask(dv7_swap, dv7);
	m8 = _mm512_cmplt_epu64_mask(dv8_swap, dv8);

	// 'AA' for the swap between adjacent lanes
	dv1 = _mm512_mask_blend_epi64(m1 ^ 0xaa, dv1_swap, dv1);
	dv2 = _mm512_mask_blend_epi64(m2 ^ 0xaa, dv2_swap, dv2);
	dv3 = _mm512_mask_blend_epi64(m3 ^ 0xaa, dv3_swap, dv3);
	dv4 = _mm512_mask_blend_epi64(m4 ^ 0xaa, dv4_swap, dv4);
	dv5 = _mm512_mask_blend_epi64(m5 ^ 0xaa, dv5_swap, dv5);
	dv6 = _mm512_mask_blend_epi64(m6 ^ 0xaa, dv6_swap, dv6);
	dv7 = _mm512_mask_blend_epi64(m7 ^ 0xaa, dv7_swap, dv7);
	dv8 = _mm512_mask_blend_epi64(m8 ^ 0xaa, dv8_swap, dv8);


	// phase 4: merge

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
	dv1_swap = _mm512_shuffle_i64x2(dv1, dv1, 0x4E);
	dv2_swap = _mm512_shuffle_i64x2(dv2, dv2, 0x4E);
	dv3_swap = _mm512_shuffle_i64x2(dv3, dv3, 0x4E);
	dv4_swap = _mm512_shuffle_i64x2(dv4, dv4, 0x4E);
	dv5_swap = _mm512_shuffle_i64x2(dv5, dv5, 0x4E);
	dv6_swap = _mm512_shuffle_i64x2(dv6, dv6, 0x4E);
	dv7_swap = _mm512_shuffle_i64x2(dv7, dv7, 0x4E);
	dv8_swap = _mm512_shuffle_i64x2(dv8, dv8, 0x4E);
	
	m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
	m2 = _mm512_cmpgt_epu64_mask(dv2, dv2_swap);
	m3 = _mm512_cmpgt_epu64_mask(dv3, dv3_swap);
	m4 = _mm512_cmpgt_epu64_mask(dv4, dv4_swap);
	m5 = _mm512_cmplt_epu64_mask(dv5, dv5_swap);
	m6 = _mm512_cmplt_epu64_mask(dv6, dv6_swap);
	m7 = _mm512_cmplt_epu64_mask(dv7, dv7_swap);
	m8 = _mm512_cmplt_epu64_mask(dv8, dv8_swap);

	// '0F' for the swap between dist-4 lanes, non alternating
	dv1 = _mm512_mask_blend_epi64(m1 ^ 0x0F, dv1_swap, dv1);
	dv2 = _mm512_mask_blend_epi64(m2 ^ 0x0F, dv2_swap, dv2);
	dv3 = _mm512_mask_blend_epi64(m3 ^ 0x0F, dv3_swap, dv3);
	dv4 = _mm512_mask_blend_epi64(m4 ^ 0x0F, dv4_swap, dv4);
	dv5 = _mm512_mask_blend_epi64(m5 ^ 0x0F, dv5_swap, dv5);
	dv6 = _mm512_mask_blend_epi64(m6 ^ 0x0F, dv6_swap, dv6);
	dv7 = _mm512_mask_blend_epi64(m7 ^ 0x0F, dv7_swap, dv7);
	dv8 = _mm512_mask_blend_epi64(m8 ^ 0x0F, dv8_swap, dv8);

	// distance 2 swaps
	dv1_swap = _mm512_permutex_epi64(dv1, 0x4E);
	dv2_swap = _mm512_permutex_epi64(dv2, 0x4E);
	dv3_swap = _mm512_permutex_epi64(dv3, 0x4E);
	dv4_swap = _mm512_permutex_epi64(dv4, 0x4E);
	dv5_swap = _mm512_permutex_epi64(dv5, 0x4E);
	dv6_swap = _mm512_permutex_epi64(dv6, 0x4E);
	dv7_swap = _mm512_permutex_epi64(dv7, 0x4E);
	dv8_swap = _mm512_permutex_epi64(dv8, 0x4E);
	
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
	dv1_swap = _mm512_shuffle_epi32(dv1, 0x4E);
	dv2_swap = _mm512_shuffle_epi32(dv2, 0x4E);
	dv3_swap = _mm512_shuffle_epi32(dv3, 0x4E);
	dv4_swap = _mm512_shuffle_epi32(dv4, 0x4E);
	dv5_swap = _mm512_shuffle_epi32(dv5, 0x4E);
	dv6_swap = _mm512_shuffle_epi32(dv6, 0x4E);
	dv7_swap = _mm512_shuffle_epi32(dv7, 0x4E);
	dv8_swap = _mm512_shuffle_epi32(dv8, 0x4E);
	
	m1 = _mm512_cmpgt_epu64_mask(dv1_swap, dv1);
	m2 = _mm512_cmpgt_epu64_mask(dv2_swap, dv2);
	m3 = _mm512_cmpgt_epu64_mask(dv3_swap, dv3);
	m4 = _mm512_cmpgt_epu64_mask(dv4_swap, dv4);
	m5 = _mm512_cmplt_epu64_mask(dv5_swap, dv5);
	m6 = _mm512_cmplt_epu64_mask(dv6_swap, dv6);
	m7 = _mm512_cmplt_epu64_mask(dv7_swap, dv7);
	m8 = _mm512_cmplt_epu64_mask(dv8_swap, dv8);

	// 'AA' for the swap between adjacent lanes
	dv1 = _mm512_mask_blend_epi64(m1 ^ 0xaa, dv1_swap, dv1);
	dv2 = _mm512_mask_blend_epi64(m2 ^ 0xaa, dv2_swap, dv2);
	dv3 = _mm512_mask_blend_epi64(m3 ^ 0xaa, dv3_swap, dv3);
	dv4 = _mm512_mask_blend_epi64(m4 ^ 0xaa, dv4_swap, dv4);
	dv5 = _mm512_mask_blend_epi64(m5 ^ 0xaa, dv5_swap, dv5);
	dv6 = _mm512_mask_blend_epi64(m6 ^ 0xaa, dv6_swap, dv6);
	dv7 = _mm512_mask_blend_epi64(m7 ^ 0xaa, dv7_swap, dv7);
	dv8 = _mm512_mask_blend_epi64(m8 ^ 0xaa, dv8_swap, dv8);


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
		dv1_swap = _mm512_shuffle_i64x2(dv1, dv1, 0x4E);
		dv2_swap = _mm512_shuffle_i64x2(dv2, dv2, 0x4E);
		dv3_swap = _mm512_shuffle_i64x2(dv3, dv3, 0x4E);
		dv4_swap = _mm512_shuffle_i64x2(dv4, dv4, 0x4E);
		dv5_swap = _mm512_shuffle_i64x2(dv5, dv5, 0x4E);
		dv6_swap = _mm512_shuffle_i64x2(dv6, dv6, 0x4E);
		dv7_swap = _mm512_shuffle_i64x2(dv7, dv7, 0x4E);
		dv8_swap = _mm512_shuffle_i64x2(dv8, dv8, 0x4E);
		
		m1 = _mm512_cmplt_epu64_mask(dv1, dv1_swap);
		m2 = _mm512_cmplt_epu64_mask(dv2, dv2_swap);
		m3 = _mm512_cmplt_epu64_mask(dv3, dv3_swap);
		m4 = _mm512_cmplt_epu64_mask(dv4, dv4_swap);
		m5 = _mm512_cmplt_epu64_mask(dv5, dv5_swap);
		m6 = _mm512_cmplt_epu64_mask(dv6, dv6_swap);
		m7 = _mm512_cmplt_epu64_mask(dv7, dv7_swap);
		m8 = _mm512_cmplt_epu64_mask(dv8, dv8_swap);

		// '0F' for the swap between dist-4 lanes, non alternating
		dv1 = _mm512_mask_blend_epi64(m1 ^ 0x0F, dv1_swap, dv1);
		dv2 = _mm512_mask_blend_epi64(m2 ^ 0x0F, dv2_swap, dv2);
		dv3 = _mm512_mask_blend_epi64(m3 ^ 0x0F, dv3_swap, dv3);
		dv4 = _mm512_mask_blend_epi64(m4 ^ 0x0F, dv4_swap, dv4);
		dv5 = _mm512_mask_blend_epi64(m5 ^ 0x0F, dv5_swap, dv5);
		dv6 = _mm512_mask_blend_epi64(m6 ^ 0x0F, dv6_swap, dv6);
		dv7 = _mm512_mask_blend_epi64(m7 ^ 0x0F, dv7_swap, dv7);
		dv8 = _mm512_mask_blend_epi64(m8 ^ 0x0F, dv8_swap, dv8);

		// distance 2 swaps
		dv1_swap = _mm512_permutex_epi64(dv1, 0x4E);
		dv2_swap = _mm512_permutex_epi64(dv2, 0x4E);
		dv3_swap = _mm512_permutex_epi64(dv3, 0x4E);
		dv4_swap = _mm512_permutex_epi64(dv4, 0x4E);
		dv5_swap = _mm512_permutex_epi64(dv5, 0x4E);
		dv6_swap = _mm512_permutex_epi64(dv6, 0x4E);
		dv7_swap = _mm512_permutex_epi64(dv7, 0x4E);
		dv8_swap = _mm512_permutex_epi64(dv8, 0x4E);
		
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
		dv1_swap = _mm512_shuffle_epi32(dv1, 0x4E);
		dv2_swap = _mm512_shuffle_epi32(dv2, 0x4E);
		dv3_swap = _mm512_shuffle_epi32(dv3, 0x4E);
		dv4_swap = _mm512_shuffle_epi32(dv4, 0x4E);
		dv5_swap = _mm512_shuffle_epi32(dv5, 0x4E);
		dv6_swap = _mm512_shuffle_epi32(dv6, 0x4E);
		dv7_swap = _mm512_shuffle_epi32(dv7, 0x4E);
		dv8_swap = _mm512_shuffle_epi32(dv8, 0x4E);
		
		m1 = _mm512_cmplt_epu64_mask(dv1_swap, dv1);
		m2 = _mm512_cmplt_epu64_mask(dv2_swap, dv2);
		m3 = _mm512_cmplt_epu64_mask(dv3_swap, dv3);
		m4 = _mm512_cmplt_epu64_mask(dv4_swap, dv4);
		m5 = _mm512_cmplt_epu64_mask(dv5_swap, dv5);
		m6 = _mm512_cmplt_epu64_mask(dv6_swap, dv6);
		m7 = _mm512_cmplt_epu64_mask(dv7_swap, dv7);
		m8 = _mm512_cmplt_epu64_mask(dv8_swap, dv8);

		// 'AA' for the swap between adjacent lanes
		dv1 = _mm512_mask_blend_epi64(m1 ^ 0xaa, dv1_swap, dv1);
		dv2 = _mm512_mask_blend_epi64(m2 ^ 0xaa, dv2_swap, dv2);
		dv3 = _mm512_mask_blend_epi64(m3 ^ 0xaa, dv3_swap, dv3);
		dv4 = _mm512_mask_blend_epi64(m4 ^ 0xaa, dv4_swap, dv4);
		dv5 = _mm512_mask_blend_epi64(m5 ^ 0xaa, dv5_swap, dv5);
		dv6 = _mm512_mask_blend_epi64(m6 ^ 0xaa, dv6_swap, dv6);
		dv7 = _mm512_mask_blend_epi64(m7 ^ 0xaa, dv7_swap, dv7);
		dv8 = _mm512_mask_blend_epi64(m8 ^ 0xaa, dv8_swap, dv8);

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
		dv1_swap = _mm512_shuffle_i64x2(dv1, dv1, 0x4E);
		dv2_swap = _mm512_shuffle_i64x2(dv2, dv2, 0x4E);
		dv3_swap = _mm512_shuffle_i64x2(dv3, dv3, 0x4E);
		dv4_swap = _mm512_shuffle_i64x2(dv4, dv4, 0x4E);
		dv5_swap = _mm512_shuffle_i64x2(dv5, dv5, 0x4E);
		dv6_swap = _mm512_shuffle_i64x2(dv6, dv6, 0x4E);
		dv7_swap = _mm512_shuffle_i64x2(dv7, dv7, 0x4E);
		dv8_swap = _mm512_shuffle_i64x2(dv8, dv8, 0x4E);
		m1 = _mm512_cmpgt_epu64_mask(dv1, dv1_swap);
		m2 = _mm512_cmpgt_epu64_mask(dv2, dv2_swap);
		m3 = _mm512_cmpgt_epu64_mask(dv3, dv3_swap);
		m4 = _mm512_cmpgt_epu64_mask(dv4, dv4_swap);
		m5 = _mm512_cmpgt_epu64_mask(dv5, dv5_swap);
		m6 = _mm512_cmpgt_epu64_mask(dv6, dv6_swap);
		m7 = _mm512_cmpgt_epu64_mask(dv7, dv7_swap);
		m8 = _mm512_cmpgt_epu64_mask(dv8, dv8_swap);

		// '0F' for the swap between dist-4 lanes, non alternating
		dv1 = _mm512_mask_blend_epi64(m1 ^ 0x0F, dv1_swap, dv1);
		dv2 = _mm512_mask_blend_epi64(m2 ^ 0x0F, dv2_swap, dv2);
		dv3 = _mm512_mask_blend_epi64(m3 ^ 0x0F, dv3_swap, dv3);
		dv4 = _mm512_mask_blend_epi64(m4 ^ 0x0F, dv4_swap, dv4);
		dv5 = _mm512_mask_blend_epi64(m5 ^ 0x0F, dv5_swap, dv5);
		dv6 = _mm512_mask_blend_epi64(m6 ^ 0x0F, dv6_swap, dv6);
		dv7 = _mm512_mask_blend_epi64(m7 ^ 0x0F, dv7_swap, dv7);
		dv8 = _mm512_mask_blend_epi64(m8 ^ 0x0F, dv8_swap, dv8);

		// distance 2 swaps
		dv1_swap = _mm512_permutex_epi64(dv1, 0x4E);
		dv2_swap = _mm512_permutex_epi64(dv2, 0x4E);
		dv3_swap = _mm512_permutex_epi64(dv3, 0x4E);
		dv4_swap = _mm512_permutex_epi64(dv4, 0x4E);
		dv5_swap = _mm512_permutex_epi64(dv5, 0x4E);
		dv6_swap = _mm512_permutex_epi64(dv6, 0x4E);
		dv7_swap = _mm512_permutex_epi64(dv7, 0x4E);
		dv8_swap = _mm512_permutex_epi64(dv8, 0x4E);
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
		dv1_swap = _mm512_shuffle_epi32(dv1, 0x4E);
		dv2_swap = _mm512_shuffle_epi32(dv2, 0x4E);
		dv3_swap = _mm512_shuffle_epi32(dv3, 0x4E);
		dv4_swap = _mm512_shuffle_epi32(dv4, 0x4E);
		dv5_swap = _mm512_shuffle_epi32(dv5, 0x4E);
		dv6_swap = _mm512_shuffle_epi32(dv6, 0x4E);
		dv7_swap = _mm512_shuffle_epi32(dv7, 0x4E);
		dv8_swap = _mm512_shuffle_epi32(dv8, 0x4E);
		m1 = _mm512_cmpgt_epu64_mask(dv1_swap, dv1);
		m2 = _mm512_cmpgt_epu64_mask(dv2_swap, dv2);
		m3 = _mm512_cmpgt_epu64_mask(dv3_swap, dv3);
		m4 = _mm512_cmpgt_epu64_mask(dv4_swap, dv4);
		m5 = _mm512_cmpgt_epu64_mask(dv5_swap, dv5);
		m6 = _mm512_cmpgt_epu64_mask(dv6_swap, dv6);
		m7 = _mm512_cmpgt_epu64_mask(dv7_swap, dv7);
		m8 = _mm512_cmpgt_epu64_mask(dv8_swap, dv8);

		// 'AA' for the swap between adjacent lanes
		dv1 = _mm512_mask_blend_epi64(m1 ^ 0xaa, dv1_swap, dv1);
		dv2 = _mm512_mask_blend_epi64(m2 ^ 0xaa, dv2_swap, dv2);
		dv3 = _mm512_mask_blend_epi64(m3 ^ 0xaa, dv3_swap, dv3);
		dv4 = _mm512_mask_blend_epi64(m4 ^ 0xaa, dv4_swap, dv4);
		dv5 = _mm512_mask_blend_epi64(m5 ^ 0xaa, dv5_swap, dv5);
		dv6 = _mm512_mask_blend_epi64(m6 ^ 0xaa, dv6_swap, dv6);
		dv7 = _mm512_mask_blend_epi64(m7 ^ 0xaa, dv7_swap, dv7);
		dv8 = _mm512_mask_blend_epi64(m8 ^ 0xaa, dv8_swap, dv8);
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

void bitonic_sort(uint64_t *data, uint32_t sz, int dir)
{
	if (sz == 64)
	{
		// base case: do the hardcoded 64-element sort
		bitonic_sort_dir_64(data, dir);
		return;
	}
	
	// two parallel half-size bitonic sorts,
	// with opposite directions.
	bitonic_sort(data, sz / 2, 0);
	bitonic_sort(data + sz / 2, sz / 2, 1);
	
	// merge in the specified direction
	bitonic_merge(data, sz, dir);
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

void parsort(uint64_t *data, uint32_t sz, int dir, int threads)
{
	// REQUIRED: threads and sz both powers of 2 and
	// (threads * 64) divides sz
	uint32_t bitonic_sort_size = sz / threads;
	uint32_t j;
	
	omp_set_num_threads(threads);

#pragma omp parallel for
	for (j = 0; j < sz / bitonic_sort_size; j++) {
		bitonic_sort(&data[j * bitonic_sort_size], bitonic_sort_size, j & 1);
	}
	
	if (threads == 1)
		return;
		
	bitonic_sort_size *= 2;
	while (bitonic_sort_size < sz)
	{
#pragma omp parallel for
		for (j = 0; j < sz / bitonic_sort_size; j++) {
			bitonic_merge(&data[j * bitonic_sort_size], bitonic_sort_size, j & 1);
		}
		bitonic_sort_size *= 2;
	}
	
	bitonic_merge(data, sz, dir);
	
	return;
}

int main(int argc, char ** argv)
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

int main1(int argc, char ** argv)
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
	
done:
	printf("sort %u x %u-bit keys in average of %lf seconds\n", 
		num_sort, key_bits, tseconds / (double)num_reps);
	
	aligned_free(loc_keys);
	return 0;
}



	