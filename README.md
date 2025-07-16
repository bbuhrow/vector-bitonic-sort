Single file, easy build, fastest SIMD-based sort of integer arrays.


The project so far contains:
1) a very fast bitonic sort routine for 64-element arrays of 64-bit integers
2) a recursive function for sorting higher-powers-of-2-sized arrays using 1) as a base case
3) a sort() function for sorting non-power-of-2 sized arrays
4) a parsort() function for sorting one large array with parallel bitonic sorts/merging (power of 2 sized arrays only, for now) using OMP
5) a test function for building N copies of a specified size and sorting each, over M repetitions (for timing), and checking for collisions between the random elements.
6) a test function for parsort()
7) items 1-3 for 32-bit arrays (128-element basecase sort/merge)
   
On an AMD Epyc 9174F processor the basecase routine will sort randomized 
  * length-64 arrays of 64-bit integers in about 77 nanoseconds (23x faster than qsort)
  * length-32k arrays of 64-bit integers in about 219 microseconds (11x faster than qsort)
  * length-1M arrays of 64-bit integers in about 12.5 milliseconds (8x faster than qsort)
  * length-128 arrays of 32-bit integers in about 107 nanoseconds (40x faster than qsort)
  * length-32k arrays of 32-bit integers in about 106 microseconds (23x faster than qsort)
  * length-1M arrays of 32-bit integers in about 6.1 milliseconds (17x faster than qsort)

REQUIRES AVX-512 instruction set support.

build with your favorite C compiler, for example:
clang -O2 -g -march=icelake-client vec_bitonic_sort.c -o vecsort

for omp support add -fopenmp

