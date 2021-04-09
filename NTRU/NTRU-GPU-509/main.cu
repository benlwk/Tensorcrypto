#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "include/params.h"


// Include local CUDA header files.
#include "include/cuda_kernel.cuh"


int main(int argc, char** argv)
{   
	uint8_t mode= 0;
  uint16_t *h_m, *h_r;
  uint8_t *h_c, *h_pk, *h_sk, *h_rm;

  cudaMallocHost((void**) &h_pk, BATCH*NTRU_PUBLICKEYBYTES* sizeof(uint8_t));
  cudaMallocHost((void**) &h_c, BATCH*NTRU_CIPHERTEXTBYTES * sizeof(uint8_t));
  cudaMallocHost((void**) &h_sk, BATCH*NTRU_SECRETKEYBYTES * sizeof(uint8_t));
  cudaMallocHost((void**) &h_rm, BATCH*NTRU_OWCPA_MSGBYTES * sizeof(uint8_t)); 
  cudaMallocHost((void**) &h_r, BATCH*NTRU_N * sizeof(uint16_t)); 
  cudaMallocHost((void**) &h_m, BATCH*NTRU_N* sizeof(uint16_t));
  
	for (int i = 1; i < argc;) {
        if (strcmp(argv[i], "-m") == 0) {
            mode = atoi(argv[i + 1]);
            i += 2;
        }
        else {           
            return 0;
        }
    }
  	
  	ntru_enc(mode, h_m, h_r, h_pk, h_c);
    ntru_dec(mode, h_rm, h_c, h_sk, h_m);
  	return 0;
}
