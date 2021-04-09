#include "../include/params.h"
#include "../include/packq.cuh"
#include <mma.h>
#include <stdio.h>
// The only dimensions currently supported by WMMA (16x16)
const int WMMA_M = 16;
#define PAD  NTRU_N%16
#define WMMA_THREAD 32

__global__ void poly_Rq_mul_gpu(uint16_t *r, uint16_t *a, uint16_t *b);
__global__ void poly_Rq_mul_gpu_shared(uint16_t *r, uint16_t *g_a, uint16_t *g_b);
__global__ void poly_Rq_mul_add_gpu(uint16_t *r, uint16_t *a, uint16_t *b, uint16_t *m);
__global__ void poly_add(uint16_t *r, uint16_t *m);
__global__ void poly_sub(uint16_t *c, uint16_t *d);
__global__ void poly_lift(uint16_t *r, const uint16_t *a);

__global__ void wmma_ker_padding(half *a, half *b, float *c);
  // Cyclic copy
__global__ void convertU16ToFp16Negatecyclic(half *out, uint16_t *in) ;
__global__ void convertU16ToFp16Negate(half *out, uint16_t *in);
__global__ void convertU16ToFp16cyclic(half *out, uint16_t *in);
__global__ void convertU16ToFp16(half *out, uint16_t *in);
__global__ void convertFp32ToU16mod2048 (uint16_t *out, float *in) ;
__global__ void remaining_gpu(uint16_t *r, uint16_t *a, uint16_t *b);
__global__ void remaining_gpu2(uint16_t *r, uint16_t *a, uint16_t *b);
__global__ void poly_Z3_to_Zq_gpu(uint16_t *r);
__global__ void poly_Rq_to_S3_gpu(uint16_t *r, uint16_t *a);
__global__ void poly_mod_3_Phi_n_gpu(uint16_t *r);
__global__ void poly_mod_q_Phi_n_gpu(uint16_t *r);
__global__ void poly_trinary_Zq_to_Z3_gpu(uint16_t *r);