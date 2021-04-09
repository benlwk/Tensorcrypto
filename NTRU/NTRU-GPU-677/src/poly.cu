
#include "../include/poly.cuh"

#include <stdlib.h> 
using namespace nvcuda;

// Must be multiples of 16 for wmma code to work
#define MODN(X) ((X) & (NTRU_N-1))   

__device__ int mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + b : r;
}

__device__ uint16_t mod3_gpu(uint16_t a)
{
  uint16_t r;
  int16_t t, c;

  r = (a >> 8) + (a & 0xff); // r mod 255 == a mod 255
  r = (r >> 4) + (r & 0xf); // r' mod 15 == r mod 15
  r = (r >> 2) + (r & 0x3); // r' mod 3 == r mod 3
  r = (r >> 2) + (r & 0x3); // r' mod 3 == r mod 3

  t = r - 3;
  c = t >> 15;

  return (c&r) ^ (~c&t);
}

__global__ void poly_Rq_mul_gpu(uint16_t *r, uint16_t *a, uint16_t *b)
{
   uint32_t i, sum;
   uint32_t tidx = threadIdx.x, bidx = blockIdx.x;

   sum = 0;// use register to accumulate
   for(i=1; i<NTRU_N-tidx; i++)
      sum += (uint32_t) a[bidx*NTRU_N + tidx+i] * (uint32_t) b[bidx*NTRU_N + NTRU_N-i];
   for(i=0; i<tidx+1; i++)
      sum += (uint32_t) a[bidx*NTRU_N + tidx-i] * (uint32_t) b[bidx*NTRU_N + i];  
   // r[bidx*NTRU_N + tidx] = sum ;
   r[bidx*NTRU_N + tidx] =MODQ(sum) ;
   // __syncthreads();
}

__global__ void poly_Rq_mul_add_gpu(uint16_t *r, uint16_t *a, uint16_t *b, uint16_t *m)
{
   uint64_t i, sum;
   uint32_t tidx = threadIdx.x, bidx = blockIdx.x;

   sum = 0;// use register to accumulate
   for(i=1; i<NTRU_N-tidx; i++)
      sum += (uint32_t) a[bidx*NTRU_N + tidx+i] * (uint32_t) b[bidx*NTRU_N + NTRU_N-i];
   for(i=0; i<tidx+1; i++)
      sum += (uint32_t) a[bidx*NTRU_N + tidx-i] * (uint32_t) b[bidx*NTRU_N + i];  
   r[bidx*NTRU_N + tidx] = sum + m[bidx*NTRU_N + tidx] ;   
   r[bidx*NTRU_N + tidx] =MODQ(r[bidx*NTRU_N + tidx] ) ;
}

__global__ void poly_Rq_mul_gpu_shared(uint16_t *r, uint16_t *g_a, uint16_t *g_b)
{
   uint16_t i, sum;
   uint32_t tidx = threadIdx.x, bidx = blockIdx.x;
   __shared__ uint16_t a[NTRU_N], b[NTRU_N];
   b[tidx] = g_b[bidx*NTRU_N + tidx];
   a[tidx] = g_a[bidx*NTRU_N + tidx];
   __syncthreads();

   sum = 0;// use register to accumulate
   for(i=0; i<tidx+1; i++)
      sum += a[tidx-i] * b[i];  
   for(i=1; i<NTRU_N-tidx; i++)
      sum += a[tidx+i] * b[(NTRU_N)-i];   
   r[bidx*NTRU_N + tidx] =MODQ(sum) ;
}


__global__ void poly_lift(uint16_t *r, const uint16_t *a)
{
  	uint32_t tid = threadIdx.x, bid = blockIdx.x;
  
    r[bid*NTRU_N + tid] = a[bid*NTRU_N + tid];  
    r[bid*NTRU_N + tid]  = r[bid*NTRU_N + tid]  | ((-(r[bid*NTRU_N + tid] >>1)) & (NTRU_Q-1));
}

__global__ void poly_add(uint16_t *r, uint16_t *m)
{
    uint32_t tid = threadIdx.x, bid = blockIdx.x;  
    r[bid*NTRU_N + tid] = r[bid*NTRU_N + tid] + m[bid*NTRU_N + tid] ;   
    r[bid*NTRU_N + tid] =MODQ(r[bid*NTRU_N + tid] ) ;
}
__global__ void poly_sub(uint16_t *c, uint16_t *d)
{
    uint32_t tid = threadIdx.x, bid = blockIdx.x;  
    c[bid*NTRU_N + tid] = c[bid*NTRU_N + tid] - d[bid*NTRU_N + tid] ;   
    // c[bid*NTRU_N + tid] =MODQ(c[bid*NTRU_N + tid] ) ;    
}

__global__ void wmma_ker_padding(half *a, half *b, float *c) {
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_M, WMMA_M, half, wmma::row_major> a_frag;
   // wklee, Read a in row major and b in column major
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_M, WMMA_M, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_M, WMMA_M, float> c_frag;

   // Each warp compute 16 elements along index i
   uint32_t warpID = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
   uint32_t ldA_offset, ldB_offset, row_idx, col_idx, st_offset;
   row_idx = warpID%((NTRU_N_PWR2)/WMMA_M)*WMMA_M;
   col_idx = warpID/((NTRU_N_PWR2)/WMMA_M)*WMMA_M;
   st_offset = row_idx + col_idx * NTRU_N_PWR2 ; 
   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0f);  
    for (int i = 0; i < (NTRU_N_PWR2)/WMMA_M; i ++)    
    {
      ldA_offset = row_idx*(NTRU_N_PWR2) + i*WMMA_M;
      ldB_offset = col_idx*(NTRU_N_PWR2) + i*WMMA_M;
    
      wmma::load_matrix_sync(a_frag, a + ldA_offset , NTRU_N_PWR2);    
      wmma::load_matrix_sync(b_frag, b + ldB_offset  , NTRU_N_PWR2);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    wmma::store_matrix_sync(c + st_offset, c_frag, NTRU_N_PWR2, wmma::mem_col_major);    
}   

// Cyclic copy
__global__ void convertU16ToFp16cyclic(half *out, uint16_t *in) {   
   int tidx = threadIdx.x;
   int bidx = blockIdx.x;

   out[bidx + tidx * (NTRU_N_PWR2)] = in[mod(tidx - bidx, NTRU_N)]; 
}

// Cyclic copy with negation
__global__ void convertU16ToFp16Negatecyclic(half *out, uint16_t *in) {   
   int tidx = threadIdx.x;
   int bidx = blockIdx.x;
  
   uint16_t temp = in[mod(tidx - bidx, NTRU_N)];
   if(temp==2047) out[bidx + tidx * (NTRU_N_PWR2)] = -1; 
   else out[bidx + tidx * (NTRU_N_PWR2)] = temp;
   
}

// Negate and copy
__global__ void convertU16ToFp16Negate(half *out, uint16_t *in) {
   int tidx = threadIdx.x;
   int bidx = blockIdx.x;
   uint16_t temp = in[bidx*NTRU_N + tidx];
   if(temp == 2047) out[bidx * (NTRU_N_PWR2) + tidx] = -1;
   else out[bidx * (NTRU_N_PWR2) + tidx] = temp;    
  // out[bidx * NTRU_N + tidx] = 99;
 
}

// Direct copy
__global__ void convertU16ToFp16(half *out, uint16_t *in) {
   int tidx = threadIdx.x;
   int bidx = blockIdx.x;

   out[bidx * (NTRU_N_PWR2) + tidx] = in[bidx * NTRU_N + tidx];      
}

// Straightforward copy
__global__ void convertFp16ToU16 (uint16_t *out, half *in) {
   int tidx = threadIdx.x;
   int bidx = blockIdx.x;

   out[bidx * NTRU_N_PWR2 + tidx] = in[bidx * NTRU_N_PWR2 + tidx];      
}

// Convert and mod
__global__ void convertFp32ToU16mod2048 (uint16_t *out, float *in) {   
   int tidx = threadIdx.x;
   int bidx = blockIdx.x;   
   int32_t tmp = (int32_t) in[bidx * (NTRU_N_PWR2) + tidx];     
   out[bidx * NTRU_N + tidx] = MODQ(tmp);   
}

/* Map {0, 1, 2} -> {0,1,q-1} in place */
__global__ void poly_Z3_to_Zq_gpu(uint16_t *r)
{    
    uint32_t tid=threadIdx.x, bid=blockIdx.x;

    r[bid*NTRU_N+ tid] = r[bid*NTRU_N+ tid] | ((-(r[bid*NTRU_N+ tid]>>1)) & (NTRU_Q-1));
}

__global__ void poly_Rq_to_S3_gpu(uint16_t *r, uint16_t *a)
{
  uint32_t tid = threadIdx.x, bid = blockIdx.x;
  uint16_t flag;

  /* The coefficients of a are stored as non-negative integers. */
  /* We must translate to representatives in [-q/2, q/2) before */
  /* reduction mod 3.                                           */
    /* Need an explicit reduction mod q here                    */
    r[bid*NTRU_N + tid] = MODQ(a[bid*NTRU_N + tid]);
    __syncthreads();
    /* flag = 1 if r[tid] >= q/2 else 0                            */
    flag = r[bid*NTRU_N + tid] >> (NTRU_LOGQ-1);
    __syncthreads();
    /* Now we will add (-q) mod 3 if r[tid] >= q/2                 */
    /* Note (-q) mod 3=(-2^k) mod 3=1<<(1-(k&1))                */
    r[bid*NTRU_N + tid] += flag << (1-(NTRU_LOGQ&1));
    __syncthreads();
  // poly_mod_3_Phi_n(r);
    r[bid*NTRU_N + tid] = mod3_gpu(r[bid*NTRU_N + tid] + 2*r[bid*NTRU_N + NTRU_N-1]);
    __syncthreads();
}

__global__ void poly_mod_3_Phi_n_gpu(uint16_t *r)
{
  uint32_t tid = threadIdx.x, bid = blockIdx.x;
  // for(i=0; i <NTRU_N; i++)
    r[bid*NTRU_N +tid] = mod3_gpu(r[bid*NTRU_N +tid] + 2*r[bid*NTRU_N +NTRU_N-1]);
}

__global__ void poly_mod_q_Phi_n_gpu(uint16_t *r)
{
  uint32_t tid = threadIdx.x, bid = blockIdx.x;
    r[bid*NTRU_N +tid] = r[bid*NTRU_N +tid] - r[NTRU_N-1];
}


/* Map {0, 1, q-1} -> {0,1,2} in place */
__global__ void poly_trinary_Zq_to_Z3_gpu(uint16_t *r)
{
  uint32_t tid=threadIdx.x, bid=blockIdx.x;
  // for(i=0; i<NTRU_N; i++)
  {
    r[bid*NTRU_N+ tid] = MODQ(r[bid*NTRU_N+ tid]);
    r[bid*NTRU_N+ tid] = 3 & (r[bid*NTRU_N+ tid] ^ (r[bid*NTRU_N+ tid]>>(NTRU_LOGQ-1)));
  }
}

