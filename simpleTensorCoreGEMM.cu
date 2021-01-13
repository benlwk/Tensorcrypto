#include <stdio.h>
#include <stdint.h>
#include <mma.h>
#include <stdlib.h> 
#include <time.h>  
#include "test_vector.h"
using namespace nvcuda;

#define MODQ(X) ((X) & (NTRU_Q-1))
#define MODN(X) ((X) & (NTRU_N_PWR2-1))   
// The only dimensions currently supported by WMMA (16x16)
const int WMMA_M = 16;

__global__ void poly_Rq_mul_gpu_shared(uint16_t *r, uint16_t *g_a, uint16_t *g_b)
{
   uint16_t i, sum;
   uint32_t tidx = threadIdx.x, bidx = blockIdx.x;
   __shared__ uint16_t a[NTRU_N], b[NTRU_N];
   b[tidx] = g_b[bidx*(NTRU_N_PWR2) + tidx];
   a[tidx] = g_a[bidx*(NTRU_N_PWR2) + tidx];
   __syncthreads();

   sum = 0;// use register to accumulate
   for(i=0; i<tidx+1; i++)
      sum += a[tidx-i] * b[i];  
   for(i=1; i<NTRU_N-tidx; i++)
      sum += a[tidx+i] * b[(NTRU_N)-i];   
   r[bidx*(NTRU_N_PWR2) + tidx] =MODQ(sum) ;
}

// This is a slower version
__global__ void poly_Rq_mul_gpu(uint16_t *r, uint16_t *a, uint16_t *b)
{
   uint16_t i, sum;
   uint32_t tidx = threadIdx.x, bidx = blockIdx.x;

   sum = 0;// use register to accumulate
   for(i=0; i<tidx+1; i++)
      sum += a[bidx*(NTRU_N_PWR2) + tidx-i] * b[bidx*(NTRU_N_PWR2) + i];  
   for(i=1; i<NTRU_N-tidx; i++)
      sum += a[bidx*(NTRU_N_PWR2) + tidx+i] * b[bidx*(NTRU_N_PWR2) + (NTRU_N)-i];   
   r[bidx*(NTRU_N_PWR2) + tidx] =MODQ(sum) ;
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

__device__ int mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + b : r;
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
   uint32_t temp = in[bidx * NTRU_N_PWR2 + tidx ];
   if(temp == 2047) out[bidx * (NTRU_N_PWR2) + tidx] = -1;
   else out[bidx * (NTRU_N_PWR2) + tidx] = temp;    
}

// Direct copy
__global__ void convertU16ToFp16(half *out, uint16_t *in) {
   int tidx = threadIdx.x;
   int bidx = blockIdx.x;

   out[bidx * (NTRU_N_PWR2) + tidx] = in[bidx * NTRU_N_PWR2 + tidx];      
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

   out[bidx * NTRU_N_PWR2 + tidx] = MODQ(tmp);   
}

int main(int argc, char* argv[]) {
   half *a_fp16, *b_fp16, *h_a_fp16, *h_b_fp16;
   float *d_a, *d_b, *d_a_fp32, *d_b_fp32, *d_c_fp32;
   float *h_c_wmma, *c_wmma ;
   uint32_t i, j, *h_c_mm_u16, *h_a_u32, *h_b_u32, *d_a_u32, *d_b_u32, *h_c_u32, *d_c_u32;
   uint16_t  *h_c_u16, *h_a_u16, *h_b_u16, *d_a_u16, *d_b_u16, *d_c_u16, *u16_c_wmma, *u16h_c_wmma, *testc;

   cudaEvent_t start, stop;
   float elapsed;
   
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
      
   cudaMalloc((void**)&a_fp16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(half));   
   cudaMalloc((void**)&b_fp16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(half));
   cudaMalloc((void**)&d_a_u32, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(uint32_t));   
   cudaMalloc((void**)&d_b_u32, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(uint32_t));    
   cudaMalloc((void**)&d_c_u32, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(uint32_t));            
   cudaMalloc((void**)&c_wmma, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(float)); 
   cudaMalloc((void**)&u16_c_wmma, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(uint16_t)); 
   cudaMalloc((void**)&d_a, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(float));     
   cudaMalloc((void**)&d_b, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(float));     
   cudaMalloc((void**)&d_a_u16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(uint16_t)); 
   cudaMalloc((void**)&d_b_u16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(uint16_t)); 
   cudaMalloc((void**)&d_c_u16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(uint16_t));     
   cudaMalloc((void**)&d_a_fp32, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(float)); 
   cudaMalloc((void**)&d_b_fp32, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(float)); 
   cudaMalloc((void**)&d_c_fp32, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(float));  

   cudaMallocHost((void**)&testc, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(uint16_t));
   cudaMallocHost((void**)&h_c_wmma, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(float));   
   cudaMallocHost((void**)&h_a_u32, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(uint32_t));   
   cudaMallocHost((void**)&h_b_u32, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(uint32_t));
   cudaMallocHost((void**)&h_c_u32, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(uint32_t));        
   cudaMallocHost((void**)&u16h_c_wmma, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(uint16_t));  
   cudaMallocHost((void**)&h_a_u16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(uint16_t)); 
   cudaMallocHost((void**)&h_b_u16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(uint16_t)); 
   cudaMallocHost((void**)&h_c_u16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(uint16_t)); 
   cudaMallocHost((void**)&h_c_mm_u16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(uint32_t));    
   cudaMallocHost((void**)&h_a_fp16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(half));   
   cudaMallocHost((void**)&h_b_fp16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(half));
   
   uint32_t t1 = NTRU_N_PWR2%WMMA_M, t2 = 0;
   if(t1!=0) t2=1;
   uint32_t threads = 32 * ((NTRU_N_PWR2+t2)/WMMA_M) * ((K+t2)/WMMA_M);// each warp computes 16x16 matrix
   uint32_t blocks = 1;
   if(threads>WMMA_THREAD) 
    {
      blocks = threads / WMMA_THREAD;
      threads = WMMA_THREAD;
    }
      
   printf("\nM = %d, PADDING = %d, K= %u, blocks: %u threads: %u \n\n", NTRU_N_PWR2, PADDING, K, blocks, threads);   

   /* initialize random seed: */   
   srand (time(NULL));  // comment out this to yield a static poly elements.
#ifdef ENCRYPT
  // This is for encryption
   for (i = 0; i <NTRU_N; i++)  
   {         
      h_a_u16[i] = MODQ(rand());   // generate constant polynomial
   }

   for (i = 0; i <NTRU_N; i++)      // generate different random vectors
      for (j = 0; j <NTRU_N; j++)    
         { h_a_u16[i * NTRU_N_PWR2 + j] = h_a_u16[j]; h_b_u16[i * NTRU_N_PWR2 + j] = MODQ(rand()%2-1); }
#else
  // This is for decryption
   for (i = 0; i <NTRU_N; i++)  
   {         
      h_a_u16[i] = MODQ(rand()%2-1);   // generate constant polynomial
   }

   for (i = 0; i <NTRU_N; i++)      // generate different random vectors
      for (j = 0; j <NTRU_N; j++)    
         { h_a_u16[i * NTRU_N_PWR2 + j] = h_a_u16[j]; h_b_u16[i * NTRU_N_PWR2 + j] = MODQ(rand()); }

#endif
 
   // printf("\na\n"); for (i = 0; i <NTRU_N_PWR2; i++) printf("%u ", h_a_u16[i]);
   // printf("\nb\n"); for (i = 0; i <NTRU_N_PWR2; i++) printf("%u ", h_b_u16[i]);
   cudaMemcpy(d_a_u16, h_a_u16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(uint16_t), cudaMemcpyDeviceToHost);
   cudaMemcpy(d_b_u16, h_b_u16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(uint16_t), cudaMemcpyDeviceToHost);
  cudaEventRecord(start); 
  printf("\nRunning with gpu u16 integer...\n");      
  for(i=0; i<REPEAT; i++)       
  {       
      // poly_Rq_mul_gpu<<<NTRU_N, NTRU_N>>>(d_c_u16, d_a_u16, d_b_u16);
    poly_Rq_mul_gpu_shared<<<K, NTRU_N>>>(d_c_u16, d_a_u16, d_b_u16);
  }
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);   
   cudaEventElapsedTime(&elapsed, start, stop);
   cudaMemcpy(h_c_u16, d_c_u16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(uint16_t), cudaMemcpyHostToDevice);       
   printf("gpu u16 took %f ms average: %.4fus\n", elapsed/REPEAT, elapsed*1000/REPEAT/K);

#ifdef DEBUG      
   printf("gpu u16 result\n"); for (i = 0; i <NTRU_N_PWR2; i++)  {
      printf("\nbatch: %u\n", i);for (j = 0; j <NTRU_N_PWR2; j++)  printf("%u ", h_c_u16[i*(NTRU_N_PWR2) + j]);  }    
#endif

   printf("\nRunning with wmma...\n");   
  cudaMemcpy(d_a_u16, h_a_u16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(int16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_b_u16, h_b_u16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(start);  
   for(i=0; i<REPEAT; i++)
   {   
#ifdef ENCRYPT           
     convertU16ToFp16Negate<<< K, NTRU_N>>>(b_fp16, d_b_u16); 
     convertU16ToFp16cyclic<<< NTRU_N, NTRU_N>>>(a_fp16, d_a_u16);
     wmma_ker_padding<<< blocks, threads >>> (a_fp16, b_fp16, c_wmma);
#else      
     convertU16ToFp16<<< NTRU_N, NTRU_N>>>(b_fp16, d_b_u16);     
     convertU16ToFp16Negatecyclic<<< NTRU_N, NTRU_N>>>(a_fp16, d_a_u16);
     wmma_ker_padding<<< blocks, threads >>> (a_fp16, b_fp16, c_wmma);
#endif
     convertFp32ToU16mod2048<<<K, NTRU_N_PWR2>>>(u16_c_wmma, c_wmma); 
   }
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   cudaMemcpy(u16h_c_wmma, u16_c_wmma, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(uint16_t), cudaMemcpyDeviceToHost);      
   float wmmaTime;
   cudaEventElapsedTime(&wmmaTime, start, stop);
   printf("gpu tensor core took %4fms average: %.4fus \n", wmmaTime/REPEAT, wmmaTime*1000/REPEAT/K);

#ifdef DEBUG   
    printf("gpu tensor core result\n"); for (int i = 0; i < NTRU_N_PWR2; i++) {
      printf("batch: %u\n", i);
      for (j = 0; j <NTRU_N_PWR2; j++) printf("%u ", u16h_c_wmma[i*(NTRU_N_PWR2) + j]);printf("\n");     }
#endif        
  printf("Speed up: %.2f\n", elapsed/wmmaTime); 

   // compare results between integer and tensor core implementation
   for (i = 0; i <NTRU_N_PWR2; i++) 
      for (j = 0; j <NTRU_N_PWR2; j++) 
      {
         if(h_c_u16[i*NTRU_N_PWR2 + j]!=u16h_c_wmma[i*NTRU_N_PWR2 + j]) 
         {
            printf("**wrong at batch %u- %u: %04x %04x\n", i, j, h_c_u16[i*NTRU_N_PWR2 + j], u16h_c_wmma[i*NTRU_N_PWR2 + j] );
            return 0;
         }
      }

  
   cudaEventDestroy(start);
   cudaEventDestroy(stop);

   cudaFree(a_fp16);
   cudaFree(b_fp16);    
   cudaFree(c_wmma);
   cudaDeviceReset();
   return 0;
}

