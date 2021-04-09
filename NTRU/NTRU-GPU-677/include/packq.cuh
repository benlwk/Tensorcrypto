#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/cuda_kernel.cuh"
__global__ void poly_Rq_sum_zero_frombytes_gpu(uint16_t *r, unsigned char *a);
__global__ void poly_Sq_tobytes_gpu(unsigned char *r, const uint16_t *a);
__global__ void poly_Rq_sum_zero_frombytes_sum_gpu(uint16_t *r);
__global__ void poly_S3_frombytes_gpu(uint16_t *r, uint8_t* msg);
__global__ void poly_S3_tobytes_gpu(uint8_t msg[NTRU_OWCPA_MSGBYTES], uint16_t *a);
__global__ void poly_Sq_frombytes_gpu_global(uint16_t *r, const unsigned char *a);