#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define LAC_Q 251	
#define LAC128	// comment out for LAC192/256

#ifdef LAC128			
#define LAC_N 512			
#else
#define LAC_N 1024
#endif

#define WMMA_THREAD LAC_N
#define REPEAT 100				// Measure average time
#define DEBUG				// Print the results of polynomial convolution
