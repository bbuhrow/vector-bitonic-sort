The project so far contains:
1) a very fast bitonic sort routine for 64-element arrays of 64-bit integers
2) a recursive function for sorting higher-powers-of-2-sized arrays using 1) as a base case
3) a test function for building N copies of a specified size and sorting each, over M repetitions (for timing), and checking for collisions between the random elements.
   
On an AMD Epyc 9174F processor the basecase routine will sort randomized 
  * length-64 arrays in about 77 nanoseconds.
  * length-32k arrays in about 219 microseconds
  * length-1M arrays in about 12.5 milliseconds

