// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Include associated header file.
#include "../include/cuda_kernel.cuh"
#include "../include/tv.h"
#include "../include/params.h"
#include "../include/packq.cuh"
#include "../include/poly.cuh"
#include "../include/check.cuh"

#include <stdio.h>
#include <stdint.h>


void ntru_enc(uint8_t mode, uint16_t *h_m, uint16_t *h_r, uint8_t *h_pk, uint8_t *h_c) {
    half *host_fp16, *h_fp16, *r_fp16;
    uint8_t *d_pk, *d_c;
    int i, j;
    uint16_t *h_h; 
    uint16_t *d_m, *d_liftm, *d_r, *d_h, *d_ct; 
    float *c_wmma;
    cudaEvent_t start, stop;
    float elapsed;
   
   /* initialize random seed: */   
    srand (time(NULL));  // comment out this to yield a static m elements.
    cudaEventCreate(&start);    cudaEventCreate(&stop);
    cudaMallocHost((void**) &h_h, BATCH*NTRU_N* sizeof(uint16_t));    
    cudaMallocHost((void**)&host_fp16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(half));   

    cudaMalloc((void**) &d_c, BATCH*NTRU_CIPHERTEXTBYTES * sizeof(uint8_t)); 
    cudaMalloc((void**) &d_ct, BATCH*NTRU_N_PWR2 * sizeof(uint16_t)); 
    cudaMalloc((void**) &d_r, BATCH*NTRU_N * sizeof(uint16_t)); 
    cudaMalloc((void**) &d_m, BATCH*NTRU_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_liftm, BATCH*NTRU_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_h, BATCH*NTRU_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_pk, BATCH*NTRU_PUBLICKEYBYTES* sizeof(uint8_t));    
    cudaMalloc((void**)&c_wmma, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(float)); 
    cudaMalloc((void**)&h_fp16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(half));   
    cudaMalloc((void**)&r_fp16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(half));    ; 

    // Using the same public-private key pair
    for(j=0; j<BATCH; j++) for(i=0; i<NTRU_PUBLICKEYBYTES; i++) h_pk[j*NTRU_PUBLICKEYBYTES + i] = pk_tv[i];
    for(j=0; j<BATCH; j++) for(i=0; i<NTRU_N; i++)  h_r [j*NTRU_N + i] = r_tv[i];
    // for(j=0; j<BATCH; j++) for(i=0; i<NTRU_N; i++)  h_m [j*NTRU_N + i] = m_tv[i];    // This is to use the same m for all encryption/decryption

    for(i=0; i<NTRU_N; i++)  h_m [i] = m_tv[i]; // From NTRU test vector
    for(j=1; j<BATCH; j++) for(i=0; i<NTRU_N-1; i++)  h_m [j*NTRU_N + i] = rand()%3;    // Randomly generated test vector

    uint32_t t1 = NTRU_N_PWR2%WMMA_M, t2 = 0;
    if(t1!=0) t2=1;
    uint32_t threads = 32 * ((NTRU_N_PWR2+t2)/WMMA_M) * ((K+t2)/WMMA_M);// each warp computes 16x16 matrix
    uint32_t blocks = 1;
    if(threads>WMMA_THREAD) 
    {
      blocks = threads / WMMA_THREAD;
      threads = WMMA_THREAD;
    }

    if(mode==0) printf("\n u16 mode N = %d, blocks: %u threads: %u \n\n", NTRU_N, blocks, threads); 
    else if(mode==1) printf("\n tensor core mode N = %d, blocks: %u threads: %u \n\n", NTRU_N, blocks, threads); 

    cudaEventRecord(start);  
    cudaMemcpy(d_pk, h_pk, BATCH*NTRU_PUBLICKEYBYTES * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, h_r, BATCH*NTRU_N*sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, BATCH*NTRU_N*sizeof(uint16_t), cudaMemcpyHostToDevice);

    // owcpa_enc
    poly_Rq_sum_zero_frombytes_gpu<<<BATCH,NTRU_PACK_DEG/8>>>(d_h, d_pk);
    poly_Rq_sum_zero_frombytes_sum_gpu<<<BATCH,1>>>(d_h);
    poly_lift<<<BATCH, NTRU_N>>>(d_liftm, d_m);
    
    if(mode==0)    
      poly_Rq_mul_gpu_shared<<<BATCH, NTRU_N>>>(d_ct, d_r, d_h);
    else if(mode==1)
    {
      convertU16ToFp16Negate<<< K, NTRU_N>>>(r_fp16, d_r); 
      convertU16ToFp16cyclic<<< NTRU_N, NTRU_N>>>(h_fp16, d_h);
      wmma_ker_padding<<< blocks, threads >>> (h_fp16, r_fp16, c_wmma);
      convertFp32ToU16mod2048<<<K, NTRU_N>>>(d_ct, c_wmma);   
    }
    poly_add<<<BATCH, NTRU_N>>>(d_ct, d_liftm);
    poly_Sq_tobytes_gpu<<<BATCH,NTRU_PACK_DEG/8>>>(d_c, d_ct); 

    // cudaMemcpy(h_h, d_ct, BATCH*NTRU_N * sizeof(uint16_t), cudaMemcpyDeviceToHost);  
    cudaMemcpy(h_c, d_c, BATCH*NTRU_CIPHERTEXTBYTES * sizeof(uint8_t), cudaMemcpyDeviceToHost);     
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);    
    cudaEventElapsedTime(&elapsed, start, stop);
    if(mode==0) printf("encrypt: gpu u16 mode took %fms average %f us\n", elapsed, elapsed*1000/NTRU_N);
    else if(mode==1) printf("encrypt: gpu tensor core mode took %fms average %f us\n", elapsed, elapsed*1000/NTRU_N);;
    
    // printf("\n h:\n"); for(j=0; j<2; j++) {printf("\nbatch: %u\n", j); for(i=0; i<NTRU_N; i++) printf("%04x ", h_h[j*NTRU_N + i]);}
    // printf("\n c:\n"); for(j=0; j<2; j++) {printf("\n\nbatch: %u\n", j); for(i=0; i<NTRU_PUBLICKEYBYTES; i++) printf("%02x ", h_c[j*NTRU_CIPHERTEXTBYTES + i]);}    
    cudaFree(d_c);  cudaFree(d_ct);  cudaFree(d_r);  cudaFree(d_m);
    cudaFree(d_liftm);  cudaFree(d_h);  cudaFree(d_pk);  cudaFree(c_wmma);
    cudaFree(h_fp16);  cudaFree(r_fp16); 
}

void ntru_dec(uint8_t mode, uint8_t *h_rm, uint8_t *h_c, uint8_t *h_sk, uint16_t *h_m) 
{
    int i, j, *fail, *d_fail;
    half *ct_fp16, *f_fp16, *host_fp16;
    uint16_t *d_ct, *d_f, *d_cf, *d_mf, *d_finv3, *d_m, *d_liftm, *d_invh, *d_r;
    uint16_t *h_h;
    uint8_t  *d_c, *d_sk, *d_rm;
    cudaEvent_t start, stop;
    float elapsed;
    float *c_wmma;

    cudaEventCreate(&start);    cudaEventCreate(&stop);
    cudaMallocHost((void**) &h_h, BATCH*NTRU_N* sizeof(uint16_t));  
    cudaMallocHost((void**)&host_fp16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(half));
    cudaMallocHost((void**)&fail, BATCH * sizeof(int));

    cudaMalloc((void**) &d_r, BATCH*NTRU_N * sizeof(uint16_t));    
    cudaMalloc((void**) &d_c, BATCH*NTRU_CIPHERTEXTBYTES * sizeof(uint8_t)); 
    cudaMalloc((void**) &d_sk, BATCH*NTRU_SECRETKEYBYTES * sizeof(uint8_t)); 
    cudaMalloc((void**) &d_ct, BATCH*NTRU_N_PWR2 * sizeof(uint16_t));    
    cudaMalloc((void**) &d_cf, BATCH*NTRU_N * sizeof(uint16_t));    
    cudaMalloc((void**) &d_f, BATCH*NTRU_N * sizeof(uint16_t));    
    cudaMalloc((void**) &d_finv3, BATCH*NTRU_N * sizeof(uint16_t));    
    cudaMalloc((void**) &d_m, BATCH*NTRU_N * sizeof(uint16_t));    
    cudaMalloc((void**) &d_rm, BATCH*NTRU_OWCPA_MSGBYTES * sizeof(uint8_t)); 
    cudaMalloc((void**) &d_mf, BATCH*NTRU_N * sizeof(uint16_t));   
    cudaMalloc((void**)&c_wmma, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(float)); 
    cudaMalloc((void**)&f_fp16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(half));   
    cudaMalloc((void**) &d_liftm, BATCH*NTRU_N* sizeof(uint16_t));    
    cudaMalloc((void**)&d_invh, NTRU_N * NTRU_N * sizeof(uint16_t));    
    cudaMalloc((void**)&ct_fp16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(half));
    cudaMalloc((void**)&d_fail, BATCH * sizeof(int));
    cudaMemset(d_invh, 0, NTRU_N * NTRU_N * sizeof(uint16_t));

   uint32_t t1 = NTRU_N_PWR2%WMMA_M, t2 = 0;
   if(t1!=0) t2=1;
   uint32_t threads = 32 * ((NTRU_N_PWR2+t2)/WMMA_M) * ((K+t2)/WMMA_M);// each warp computes 16x16 matrix
   uint32_t blocks = 1;
   if(threads>WMMA_THREAD) 
    {
      blocks = threads / WMMA_THREAD;
      threads = WMMA_THREAD;
    }

    // Using the same public-private key pair
    for(j=0; j<BATCH; j++) for(i=0; i<NTRU_SECRETKEYBYTES; i++)  h_sk [j*NTRU_SECRETKEYBYTES + i] = sk_tv[i];

    cudaEventRecord(start);
    cudaMemcpy(d_c, h_c, BATCH*NTRU_CIPHERTEXTBYTES*sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sk, h_sk, BATCH*NTRU_SECRETKEYBYTES*sizeof(uint8_t), cudaMemcpyHostToDevice);
    poly_Rq_sum_zero_frombytes_gpu<<<BATCH,NTRU_PACK_DEG/8>>>(d_ct, d_c);
    poly_Rq_sum_zero_frombytes_sum_gpu<<<BATCH,1>>>(d_ct);
    poly_S3_frombytes_gpu<<<BATCH, NTRU_N>>>(d_f, d_sk);    
    poly_Z3_to_Zq_gpu<<<BATCH, NTRU_N>>>(d_f);
    if(mode==0)
    {        
        poly_Rq_mul_gpu_shared<<<BATCH, NTRU_N>>>(d_cf, d_ct, d_f);
    } 
    else if(mode==1)
    {
      convertU16ToFp16<<< K, NTRU_N>>>(ct_fp16, d_ct); 
      convertU16ToFp16Negatecyclic<<< NTRU_N, NTRU_N>>>(f_fp16, d_f);
      wmma_ker_padding<<< blocks, threads >>> (f_fp16, ct_fp16, c_wmma);    
      convertFp32ToU16mod2048<<<K, NTRU_N>>>(d_cf, c_wmma);          
    }
    poly_Rq_to_S3_gpu<<<BATCH, NTRU_N>>>(d_mf, d_cf);
    poly_S3_frombytes_gpu<<<BATCH, NTRU_N>>>(d_finv3, d_sk+NTRU_PACK_TRINARY_BYTES);    

    // //poly_S3_mul
    if(mode==0)
      poly_Rq_mul_gpu_shared<<<BATCH, NTRU_N>>>(d_m, d_finv3, d_mf);
    else if(mode==1)
    {      
      convertU16ToFp16<<< K, NTRU_N>>>(f_fp16, d_mf);
      convertU16ToFp16cyclic<<< NTRU_N, NTRU_N>>>(ct_fp16, d_finv3);
      wmma_ker_padding<<< blocks, threads >>> (ct_fp16, f_fp16, c_wmma);    
      convertFp32ToU16mod2048<<<K, NTRU_N>>>(d_m, c_wmma); 
    } 
    poly_mod_3_Phi_n_gpu<<<BATCH, NTRU_N>>>(d_m);    
    poly_S3_tobytes_gpu<<<BATCH, NTRU_PACK_DEG/5>>>(d_rm+NTRU_PACK_TRINARY_BYTES, d_m);
    owcpa_check_ciphertext_gpu<<<BATCH, 1>>>(d_fail, d_c);
    owcpa_check_m_gpu<<<BATCH, 1>>>(d_fail, d_m);
    //   /* b = c - Lift(m) mod (q, x^n - 1) */
    poly_lift<<<BATCH, NTRU_N>>>(d_liftm, d_m);
    poly_sub<<<BATCH, NTRU_N>>>(d_ct, d_liftm);    
    poly_Sq_frombytes_gpu_global<<<BATCH, NTRU_PACK_DEG/8>>>(d_invh, d_sk+2*NTRU_PACK_TRINARY_BYTES);

    // --> this part cannot use tensor core because both poly are dense, it can overflow the FP32 accumulator in tensor core
    // //poly_Sq_mul(r, b, invh); 
    poly_Rq_mul_gpu_shared<<<BATCH, NTRU_N>>>(d_r, d_ct, d_invh);
    poly_mod_q_Phi_n_gpu<<<BATCH, NTRU_N>>>(d_r);
    owcpa_check_r_gpu<<<BATCH, 1>>>(d_fail, d_r);
    poly_trinary_Zq_to_Z3_gpu<<<BATCH, NTRU_N>>>(d_r);
    poly_S3_tobytes_gpu<<<BATCH, NTRU_PACK_DEG/5>>>(d_rm, d_r);
#ifdef DEBUG    
    cudaMemcpy(h_h, d_m, BATCH*NTRU_N * sizeof(uint16_t), cudaMemcpyDeviceToHost);    
#endif    
    cudaMemcpy(h_rm, d_rm, BATCH*NTRU_OWCPA_MSGBYTES * sizeof(uint8_t), cudaMemcpyDeviceToHost);    
    // cudaMemcpy(host_fp16, f_fp16, NTRU_N_PWR2 * NTRU_N_PWR2 * sizeof(half),      cudaMemcpyDeviceToHost);   
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);    
    cudaEventElapsedTime(&elapsed, start, stop);
    if(mode==0) printf("\ndecrypt: gpu u16 mode took %fms average %f us\n", elapsed, elapsed*1000/NTRU_N);
    else if(mode==1) printf("\ndecrypt: gpu tensor core mode took %fms average %f us\n", elapsed, elapsed*1000/NTRU_N);;       

#ifdef DEBUG
    // Compare with the plaintext h_m
    for(j=0; j<BATCH; j++) for(i=0; i<NTRU_N; i++) 
    {
        if(h_h[j*NTRU_N + i] != h_m[j*NTRU_N + i])
        {
            printf("Wrong at batch %u element %u: %u %u\n", j, i, h_m[j*NTRU_N + i], h_h[j*NTRU_N + i]);
        }
    }

    // printf("\n h:\n"); for(j=0; j<2; j++) {printf("\nbatch: %u\n", j); for(i=0; i<NTRU_N; i++) printf("%04x ", h_h[j*NTRU_N + i]);}
    printf("\n rm:\n"); for(j=0; j<2; j++) {printf("\nbatch: %u\n", j); for(i=0; i<NTRU_OWCPA_MSGBYTES; i++) printf("%02x ", h_rm[j*NTRU_OWCPA_MSGBYTES + i]);}
#endif
    cudaFree(d_c);  cudaFree(d_ct);  cudaFree(d_r);  cudaFree(d_sk);
    cudaFree(d_liftm);  cudaFree(d_m);  cudaFree(d_cf);  cudaFree(c_wmma);
    cudaFree(d_rm);  cudaFree(d_f); cudaFree(d_finv3);    cudaFree(d_mf); 
    cudaFree(f_fp16);   cudaFree(d_invh);   cudaFree(ct_fp16); 
    cudaFreeHost(h_h);  cudaFreeHost(host_fp16);
}