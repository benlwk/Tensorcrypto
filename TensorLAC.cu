#include <stdio.h>
#include <stdint.h>
#include <mma.h>
#include <stdlib.h> 
#include <time.h>  
#include "params.h"
using namespace nvcuda;

#define BIG_Q 1024*LAC_Q//1024*Q 
#define MODQ(X) (X+BIG_Q)%LAC_Q
#define MODN(X) ((X) & (LAC_N-1))   
// The only dimensions currently supported by WMMA (16x16)
const int WMMA_M = 16;

// This is from the reference implementation
__global__ void poly_mul_ref_gpu(const uint8_t *a, const uint8_t *s, uint8_t *b)
{  
  // __shared__ unsigned char v[LAC_N+LAC_N];  
  __shared__ unsigned char v[LAC_N+LAC_N];  
  int32_t sum, j;
  uint32_t tid = threadIdx.x, bid = blockIdx.x;

  //construct matrix of a
  v[tid]=a[bid*LAC_N + LAC_N-1-tid];
  v[tid+LAC_N]=LAC_Q-v[tid];
  __syncthreads();

  sum=0;
  for(j=0;j<LAC_N;j++)
    sum+= (int32_t)v[LAC_N-tid-1 + j]*(int32_t)s[bid*LAC_N + j];

  b[bid*LAC_N + tid]=MODQ(sum);
}

// This seems to be slower
__global__ void poly_Rq_mul_gpu_shared_u8(uint8_t *r, uint8_t *g_a, uint8_t *g_b)
{
   int32_t i, sum;   
   uint32_t tidx = threadIdx.x, bidx = blockIdx.x;
   __shared__ uint16_t a[LAC_N], b[LAC_N];
   b[tidx] = g_b[bidx*(LAC_N) + tidx];
   a[tidx] = g_a[bidx*(LAC_N) + tidx];
   __syncthreads();

   sum = 0;// use register to accumulate
   for(i=0; i<tidx+1; i++)
      sum += a[tidx-i] * b[i];  
   for(i=1; i<LAC_N-tidx; i++)
      sum -= a[tidx+i] * b[(LAC_N)-i];       
   sum = MODQ(sum);   
   if(sum<0) sum+=LAC_Q;
   r[bidx*(LAC_N) + tidx] = sum  ;
}
 

// uint8_t version
__global__ void wmma_ker_u8(uint8_t *a, uint8_t *b, int *c) {
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_M, WMMA_M, uint8_t, wmma::row_major> a_frag;
   // wklee, Read a in row major and b in column major
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_M, WMMA_M, uint8_t, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_M, WMMA_M, int> c_frag;

   // Each warp compute 16 elements along index i
   uint32_t warpID = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
   uint32_t ldA_offset, ldB_offset, row_idx, col_idx, st_offset;
   row_idx = warpID%((LAC_N)/WMMA_M)*WMMA_M;
   col_idx = warpID/((LAC_N)/WMMA_M)*WMMA_M;
   st_offset = row_idx + col_idx * LAC_N ; 
   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0f);  
    for (int i = 0; i < (LAC_N)/WMMA_M; i ++)    
    {
      ldA_offset = row_idx*(LAC_N) + i*WMMA_M;
      ldB_offset = col_idx*(LAC_N) + i*WMMA_M;
    
      wmma::load_matrix_sync(a_frag, a + ldA_offset , LAC_N);    
      wmma::load_matrix_sync(b_frag, b + ldB_offset  , LAC_N);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    wmma::store_matrix_sync(c + st_offset, c_frag, LAC_N, wmma::mem_col_major);    
}  

__device__ int mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + b : r;
}

__global__ void convertU8ToU8cyclic(uint8_t *out, uint8_t *in) {   
   int tidx = threadIdx.x;
   int bidx = blockIdx.x;

   if(tidx - bidx<0) 
    out[bidx + tidx * LAC_N] = LAC_Q-in[mod(tidx - bidx, LAC_N)]; 
   else
    out[bidx + tidx * (LAC_N)] = in[mod(tidx - bidx, LAC_N)]; 
}

// Convert and mod
__global__ void convertINT32ToU8modQ (uint8_t *out, int *in) {   
   int tidx = threadIdx.x;
   int bidx = blockIdx.x;   
   int32_t tmp = (int32_t) in[bidx * (LAC_N) + tidx];     
   tmp = MODQ(tmp);
   // if(tmp<0) tmp+=LAC_Q;
   out[bidx * LAC_N + tidx] = tmp;   
}



int main(int argc, char* argv[]) {      
   int *d_c_wmma_u8;
   uint32_t i, j;

   uint8_t *h_a, *h_s, *h_b, *h_b_gpu_u8;
   uint8_t *d_a_u8, *d_s_u8, *d_b_u8, *d_a_u8_cyclic, *d_c_u8;

   cudaEvent_t start, stop;
   float elapsed;
   
   cudaEventCreate(&start);
   cudaEventCreate(&stop);   
 
   cudaMalloc((void**) &d_a_u8, LAC_N*LAC_N* sizeof(uint8_t));
   cudaMalloc((void**) &d_s_u8, LAC_N*LAC_N* sizeof(uint8_t));
   cudaMalloc((void**) &d_a_u8_cyclic, LAC_N*LAC_N* sizeof(uint8_t));
   cudaMalloc((void**) &d_b_u8, LAC_N*LAC_N* sizeof(uint8_t));
   cudaMalloc((void**) &d_c_u8, LAC_N*LAC_N* sizeof(uint8_t));
   cudaMalloc((void**) &d_c_wmma_u8, LAC_N*LAC_N* sizeof(int));   
  
  cudaMallocHost((void**) &h_a, LAC_N*LAC_N* sizeof(uint8_t));
  cudaMallocHost((void**) &h_s, LAC_N*LAC_N* sizeof(uint8_t));
  cudaMallocHost((void**) &h_b, LAC_N*LAC_N* sizeof(uint8_t));  
  cudaMallocHost((void**) &h_b_gpu_u8, LAC_N*LAC_N* sizeof(uint8_t));
   
   uint32_t t1 = LAC_N%WMMA_M, t2 = 0;
   if(t1!=0) t2=1;
   uint32_t threads = 32 * ((LAC_N+t2)/WMMA_M) * ((LAC_N+t2)/WMMA_M);// each warp computes 16x16 matrix
   uint32_t blocks = 1;
   if(threads>WMMA_THREAD) 
    {
      blocks = threads / WMMA_THREAD;
      threads = WMMA_THREAD;
    }
      
   printf("\nM = %d, LAC_N= %u, blocks: %u threads: %u \n\n", LAC_N, LAC_N, blocks, threads);   

   /* initialize random seed: */   
   srand (time(NULL));  // comment out this to yield a static poly elements.
  // This is for encryption
   for (i = 0; i <LAC_N; i++)  
   {         
      h_a[i] = MODQ(rand());   // generate constant polynomial    
   }

   for (i = 0; i <LAC_N; i++)      // generate different random vectors
      for (j = 0; j <LAC_N; j++)    
        {           
          h_a[i * LAC_N + j] = h_a[j]; 
          h_s[i * LAC_N + j] = MODQ(rand()%3-1);  // between -1 to +1
        }
 
   // printf("\na\n"); for (i = 0; i <LAC_N; i++) printf("%u ", h_a_u16[i]);
   // printf("\nb\n"); for (i = 0; i <LAC_N; i++) printf("%u ", h_b_u16[i]);
  cudaMemcpy(d_a_u8, h_a, LAC_N * LAC_N * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(d_s_u8, h_s, LAC_N * LAC_N * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  printf("\nRunning with gpu u32 integer unit...\n");      
  cudaEventRecord(start);   
  for(i=0; i<REPEAT; i++)       
  {         
    poly_Rq_mul_gpu_shared_u8<<<LAC_N, LAC_N>>>(d_b_u8, d_a_u8, d_s_u8);
    // poly_mul_ref_gpu<<<LAC_N, LAC_N>>>(d_a_u8, d_s_u8, d_b_u8); 
  }
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);   
   cudaEventElapsedTime(&elapsed, start, stop);     
   cudaMemcpy(h_b, d_b_u8, LAC_N * LAC_N * sizeof(uint8_t), cudaMemcpyHostToDevice);       
   printf("gpu u32 took %f ms\n", elapsed/REPEAT);

#ifdef DEBUG      
   printf("gpu u16 result\n"); for (i = 0; i <2; i++)  {
      printf("\nbatch: %u\n", i);for (j = 0; j <LAC_N; j++)  printf("%u ",  h_b[i*(LAC_N) + j] );  }    
#endif

   printf("\nRunning with wmma...\n");   
   cudaEventRecord(start);  
   for(i=0; i<REPEAT; i++)
   {       
     convertU8ToU8cyclic<<< LAC_N, LAC_N>>>(d_a_u8_cyclic, d_a_u8);
     wmma_ker_u8<<< blocks, threads >>> (d_a_u8_cyclic, d_s_u8, d_c_wmma_u8);
     convertINT32ToU8modQ<<<LAC_N, LAC_N>>>(d_c_u8, d_c_wmma_u8);
   }
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
 
    cudaMemcpy(h_b_gpu_u8, d_c_u8, LAC_N * LAC_N * sizeof(uint8_t), cudaMemcpyDeviceToHost);      
   float wmmaTime;
   cudaEventElapsedTime(&wmmaTime, start, stop);
   printf("gpu tensor core took %fms\n", wmmaTime/REPEAT);

#ifdef DEBUG   
    printf("gpu tensor core result\n"); for (int i = 0; i < 2; i++) {
      printf("batch: %u\n", i);
      for (j = 0; j <LAC_N; j++) printf("%u ", h_b_gpu_u8[i*(LAC_N) + j]);printf("\n");     }
#endif        
  printf("Speed up: %.2f\n", elapsed/wmmaTime); 

   // compare results between integer and tensor core implementation
   for (i = 0; i <LAC_N; i++) 
      for (j = 0; j <LAC_N; j++) 
      {
         // if(h_b[i*LAC_N + j]!=u16h_c_wmma[i*LAC_N + j]) 
        if(h_b[i*LAC_N + j]!=h_b_gpu_u8[i*LAC_N + j]) 
        {
            printf("**wrong at batch %u- %u: %u %u\n", i, j, h_b[i*LAC_N + j], h_b_gpu_u8[i*LAC_N + j] );
            return 0;
         }
      }

  
   cudaEventDestroy(start);
   cudaEventDestroy(stop);

   cudaFree(d_a_u8);
   cudaFree(d_s_u8);
   cudaFree(d_b_u8);
   cudaFree(d_a_u8_cyclic);
   cudaFree(d_c_u8);
   cudaFreeHost(h_a);
   cudaFreeHost(h_s);
   cudaFreeHost(h_b);
   cudaFreeHost(h_b_gpu_u8);
   cudaDeviceReset();
   return 0;
}

