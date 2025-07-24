Single file, easy build, very fast SIMD-based sort of integer arrays.

The speed comes from a set of highly unrolled base-case functions 
that sort chunks of data entirely in-core, which hides instruction latency and 
minimizes data I/O.  Also, it appears to deviate from many other AVX-512 bitonic 
sorts in that it uses the following set of intra-vector swap intrinsics:

#define SWAP16(x) _mm512_rol_epi32((x), 16)
#define SWAP32(x) _mm512_shuffle_epi32((x), 0xB1)
#define SWAP64(x) _mm512_shuffle_epi32((x), 0x4E)
#define SWAP128(x) _mm512_permutex_epi64((x), 0x4E)
#define SWAP256(x) _mm512_shuffle_i64x2((x), (x), 0x4E)

As opposed to _mm512_permutexvar_epi*, which requires a separate index register
for the permutation indices.  Using immediates allows all of the registers
to be used for data and sidesteps needing to load the index registers.

The 64-bit base case sort was found to be faster using a cmp/blend combination,
instead of the min/max approach used by 16 and 32-bit sorts.

The base-case functions and their performance relative to qsort are:
bitonic_sort16_128: sort 128 2-byte integers (256 bytes) in 54 nanoseconds (101x faster than qsort)
bitonic_sort16_256: sort 256 2-byte integers (512 bytes) in 109 nanoseconds (119x faster than qsort)
bitonic_sort32_64 : sort 64 4-byte integers (256 bytes) in tbd nanoseconds (tbd faster than qsort)
bitonic_sort32_128: sort 128 4-byte integers (512 bytes) in 85 nanoseconds (51x faster than qsort)
bitonic_sort32_256: sort 256 4-byte integers (1024 bytes) in 187 nanoseconds (53x faster than qsort)
bitonic_sort64_32 : sort 32 8-byte integers (256 bytes) in tbd nanoseconds (tbd faster than qsort)
bitonic_sort64_64 : sort 64 8-byte integers (512 bytes) in 77 nanoseconds (23x faster than qsort)
bitonic_sort64_128: sort 128 8-byte integers (1024 bytes) in tbd nanoseconds (tbd faster than qsort)

Other functions:
bitonic_sort(): a recursive function for sorting higher-powers-of-2-sized arrays
sort()        : a function for sorting arbitrary sized arrays
parsort()     : a function for parallel sorting of higher-powers-of-2-sized arrays using openMP
   
None of the sort routines detect or care about how the input data is distributed.  Random
or already sorted, the speed will be the same.  Here are a few benchmarks of longer lists:

length-32k arrays of 64-bit integers  in about 219 microseconds		(11x faster than qsort)
length-1M arrays of 64-bit integers  in about 12.5 milliseconds		(8x faster than qsort)
length-32k arrays of 32-bit integers  in about 90 microseconds		(27x faster than qsort)
length-1M arrays of 32-bit integers  in about 5.4 milliseconds		(19x faster than qsort)
length-32k arrays of 16-bit integers  in about 46 microseconds		(69x faster than qsort)
length-1M arrays of 16-bit integers  in about 2.7 milliseconds		(48x faster than qsort)

All benchmarks were run on an AMD Epyc 9174F processor.

pure C build, for example:
clang -O2 -g -march=icelake-client -fopenmp vec_bitonic_sort.c -o vecsort
gcc -O2 -g -march=icelake-client -fopenmp vec_bitonic_sort.c -o vecsort
icc -O2 -g -march=icelake-client -fopenmp vec_bitonic_sort.c -o vecsort

Future plans (in progress):
* sorts for 32-bit floats
* sorts for 64-bit doubles
* general sorts (i.e., array of structures with keys)


