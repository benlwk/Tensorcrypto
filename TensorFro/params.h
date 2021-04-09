#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define FRO_Q 2048		
	
// #define FRODO_II		// Select one of these two
#define TENSORFRO

#define CLIENT			// Select client or server
// #define SERVER

#ifdef FRODO_II
#define FRO_N_PWR2 576	// Multiple of 32
#define PADDING 6
#define FRO_N FRO_N_PWR2 - PADDING
#ifdef CLIENT
#define K 512		
#else
#define K 570		
#endif
#endif

#ifdef TENSORFRO
#define FRO_N_PWR2 576	// Multiple of 32
#define PADDING 16
#define FRO_N FRO_N_PWR2 - PADDING
#ifdef CLIENT
#define K 552		
#else
#define K 550		
#endif
#endif

#define WMMA_THREAD FRO_N_PWR2
#define REPEAT 1
#define DEBUG