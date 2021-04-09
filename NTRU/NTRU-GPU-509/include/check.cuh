#include <stdio.h>
#include <stdint.h>
#include "../include/cuda_kernel.cuh"
#include "../include/params.h"

__global__ void owcpa_check_ciphertext_gpu(int *fail, uint8_t *ciphertext);

__global__ void owcpa_check_r_gpu(int *fail, uint16_t *r);

__global__ void owcpa_check_m_gpu(int *fail, uint16_t *m);
