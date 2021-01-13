#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define NTRU_Q 2048				// For hps2048509 and hps2048677
// #define NTRUHPS2048509			// Select only one of these two
#define NTRUHPS2048677

#define ENCRYPT

#ifdef NTRUHPS2048509
#define NTRU_N_PWR2 512			// Multiple of 32
#define PADDING 3
#define K NTRU_N_PWR2			// Multiple of 16
#define NTRU_N NTRU_N_PWR2 - PADDING
#endif

#ifdef NTRUHPS2048677
#define NTRU_N_PWR2 704			// Multiple of 32
#define PADDING 27
#define K NTRU_N_PWR2			// Multiple of 16
#define NTRU_N NTRU_N_PWR2 - PADDING
#endif

#define WMMA_THREAD NTRU_N_PWR2
#define REPEAT 100				// Measure average time
// #define DEBUG				// Print the results of polynomial convolution
