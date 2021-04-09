#include "../include/params.h"
#include "../include/packq.cuh"
#include <stdio.h>

__device__ void poly_Sq_frombytes_gpu(uint16_t *r, const unsigned char *a)
{
  int i = NTRU_PACK_DEG/8;
	uint32_t tid = threadIdx.x, bid = blockIdx.x;
  uint32_t b_offset = bid*NTRU_N, a_offset = bid*NTRU_PUBLICKEYBYTES;

    r[b_offset + 8*tid+0] = (a[a_offset + 11*tid+ 0] >> 0) | (((uint16_t)a[a_offset + 11*tid+ 1] & 0x07) << 8);
    r[b_offset + 8*tid+1] = (a[a_offset + 11*tid+ 1] >> 3) | (((uint16_t)a[a_offset + 11*tid+ 2] & 0x3f) << 5);
    r[b_offset + 8*tid+2] = (a[a_offset + 11*tid+ 2] >> 6) | (((uint16_t)a[a_offset + 11*tid+ 3] & 0xff) << 2) | (((uint16_t)a[a_offset + 11*tid+ 4] & 0x01) << 10);
    r[b_offset + 8*tid+3] = (a[a_offset + 11*tid+ 4] >> 1) | (((uint16_t)a[a_offset + 11*tid+ 5] & 0x0f) << 7);
    r[b_offset + 8*tid+4] = (a[a_offset + 11*tid+ 5] >> 4) | (((uint16_t)a[a_offset + 11*tid+ 6] & 0x7f) << 4);
    r[b_offset + 8*tid+5] = (a[a_offset + 11*tid+ 6] >> 7) | (((uint16_t)a[a_offset + 11*tid+ 7] & 0xff) << 1) | (((uint16_t)a[a_offset + 11*tid+ 8] & 0x03) <<  9);
    r[b_offset + 8*tid+6] = (a[a_offset + 11*tid+ 8] >> 2) | (((uint16_t)a[a_offset + 11*tid+ 9] & 0x1f) << 6);
    r[b_offset + 8*tid+7] = (a[a_offset + 11*tid+ 9] >> 5) | (((uint16_t)a[a_offset + 11*tid+10] & 0xff) << 3);
  
  if(tid==0)	//wklee, can be parallelized later
  {
	  switch(NTRU_PACK_DEG&0x07)
	  {
	    // cases 0 and 6 are impossible since 2 generates (Z/n)* and
	    // p mod 8 in {1, 7} implies that 2 is a quadratic residue.
	    case 4:
	      r[b_offset + 8*i+0] = (a[a_offset + 11*i+ 0] >> 0) | (((uint16_t)a[a_offset + 11*i+ 1] & 0x07) << 8);
	      r[b_offset + 8*i+1] = (a[a_offset + 11*i+ 1] >> 3) | (((uint16_t)a[a_offset + 11*i+ 2] & 0x3f) << 5);
	      r[b_offset + 8*i+2] = (a[a_offset + 11*i+ 2] >> 6) | (((uint16_t)a[a_offset + 11*i+ 3] & 0xff) << 2) | (((uint16_t)a[a_offset + 11*i+ 4] & 0x01) << 10);
	      r[b_offset + 8*i+3] = (a[a_offset + 11*i+ 4] >> 1) | (((uint16_t)a[a_offset + 11*i+ 5] & 0x0f) << 7);
	      break;
	    case 2:
	      r[b_offset + 8*i+0] = (a[a_offset + 11*i+ 0] >> 0) | (((uint16_t)a[a_offset + 11*i+ 1] & 0x07) << 8);
	      r[b_offset + 8*i+1] = (a[a_offset + 11*i+ 1] >> 3) | (((uint16_t)a[a_offset + 11*i+ 2] & 0x3f) << 5);
	      break;
	  }
	  r[NTRU_N-1] = 0;
	}
}

__global__ void poly_Sq_tobytes_gpu(unsigned char *r, const uint16_t *a)
{
    int i,j;
    uint16_t t[8];
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    uint32_t b_offset = bid*NTRU_CIPHERTEXTBYTES, a_offset = bid*NTRU_N;

    for(j=0;j<8;j++)// wklee, we can remove this, already MOD
      t[j] = MODQ(a[a_offset + 8*tid+j]);

    r[b_offset + 11 * tid + 0] = (unsigned char) ( t[0]        & 0xff);
    r[b_offset + 11 * tid + 1] = (unsigned char) ((t[0] >>  8) | ((t[1] & 0x1f) << 3));
    r[b_offset + 11 * tid + 2] = (unsigned char) ((t[1] >>  5) | ((t[2] & 0x03) << 6));
    r[b_offset + 11 * tid + 3] = (unsigned char) ((t[2] >>  2) & 0xff);
    r[b_offset + 11 * tid + 4] = (unsigned char) ((t[2] >> 10) | ((t[3] & 0x7f) << 1));
    r[b_offset + 11 * tid + 5] = (unsigned char) ((t[3] >>  7) | ((t[4] & 0x0f) << 4));
    r[b_offset + 11 * tid + 6] = (unsigned char) ((t[4] >>  4) | ((t[5] & 0x01) << 7));
    r[b_offset + 11 * tid + 7] = (unsigned char) ((t[5] >>  1) & 0xff);
    r[b_offset + 11 * tid + 8] = (unsigned char) ((t[5] >>  9) | ((t[6] & 0x3f) << 2));
    r[b_offset + 11 * tid + 9] = (unsigned char) ((t[6] >>  6) | ((t[7] & 0x07) << 5));
    r[b_offset + 11 * tid + 10] = (unsigned char) ((t[7] >>  3));
    __syncthreads();
  if(tid==0)//wklee, can be parallelized later
  {
    i = 84;
    // wklee, we can remove this, already MOD
    for(j=0;j<NTRU_PACK_DEG-8*i;j++)    t[j] = MODQ(a[bid*NTRU_N + 8*i+j]);
    for(; j<8; j++)
    t[j] = 0;
    switch(NTRU_PACK_DEG&0x07)
    {
      // cases 0 and 6 are impossible since 2 generates (Z/n)* and
      // p mod 8 in {1, 7} implies that 2 is a quadratic residue.
      case 4:
        r[b_offset + 11 * i + 0] = (unsigned char) (t[0]        & 0xff);
        r[b_offset + 11 * i + 1] = (unsigned char) (t[0] >>  8) | ((t[1] & 0x1f) << 3);
        r[b_offset + 11 * i + 2] = (unsigned char) (t[1] >>  5) | ((t[2] & 0x03) << 6);
        r[b_offset + 11 * i + 3] = (unsigned char) (t[2] >>  2) & 0xff;
        r[b_offset + 11 * i + 4] = (unsigned char) (t[2] >> 10) | ((t[3] & 0x7f) << 1);
        r[b_offset + 11 * i + 5] = (unsigned char) (t[3] >>  7) | ((t[4] & 0x0f) << 4);
        break;
      case 2:
        r[b_offset + 11 * i + 0] = (unsigned char) (t[0]        & 0xff);
        r[b_offset + 11 * i + 1] = (unsigned char) (t[0] >>  8) | ((t[1] & 0x1f) << 3);
        r[b_offset + 11 * i + 2] = (unsigned char) (t[1] >>  5) | ((t[2] & 0x03) << 6);
        break;
    }
  }
}


__global__ void poly_Rq_sum_zero_frombytes_gpu(uint16_t *r, uint8_t *a)
{
  uint32_t tid = threadIdx.x, bid = blockIdx.x;
  uint32_t b_offset = bid*NTRU_N;
  int i;
  poly_Sq_frombytes_gpu(r,a);
  // __syncthreads();
  // /* Set r[n-1] so that the sum of coefficients is zero mod q */  
  // if(tid==0) 
  // {
  //   // for(i=0;i<NTRU_PACK_DEG;i++) r[b_offset + NTRU_N-1] += r[i];   
  //    // r[b_offset + NTRU_N-1] = MODQ(-(r[b_offset + NTRU_N-1]));
  // }
}

__global__ void poly_Rq_sum_zero_frombytes_sum_gpu(uint16_t *r)
{
    int i;
    uint32_t b_offset = blockIdx.x*NTRU_N;
    /* Set r[n-1] so that the sum of coefficients is zero mod q */  
    for(i=0;i<NTRU_PACK_DEG;i++) r[b_offset + NTRU_N-1] += r[i];       
    r[b_offset + NTRU_N-1] = MODQ(-(r[b_offset + NTRU_N-1]));
}

__device__ uint16_t mod3(uint16_t a)
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

__global__ void poly_S3_frombytes_gpu(uint16_t *r, uint8_t* msg)
{
  int i, j;
  uint8_t c;
  uint32_t tid = threadIdx.x, bid = blockIdx.x;

  if(tid < NTRU_PACK_DEG/5)
  {
    c = msg[bid*NTRU_SECRETKEYBYTES + tid];
    r[bid*NTRU_N + 5*tid+0] = c;
    r[bid*NTRU_N + 5*tid+1] = c * 171 >> 9;  // this is division by 3
    r[bid*NTRU_N + 5*tid+2] = c * 57 >> 9;  // division by 3^2
    r[bid*NTRU_N + 5*tid+3] = c * 19 >> 9;  // division by 3^3
    r[bid*NTRU_N + 5*tid+4] = c * 203 >> 14;  // etc.
  }
  if(tid==0)
  {
      i = NTRU_PACK_DEG/5;
      c = msg[bid*NTRU_SECRETKEYBYTES +i];
      for(j=0; (5*i+j)<NTRU_PACK_DEG; j++)
      {
        r[bid*NTRU_N +5*i+j] = c;
        c = c * 171 >> 9;
      }
      r[bid*NTRU_N +NTRU_N-1] = 0;
  }
  __syncthreads();
  // poly_mod_3_Phi_n(r);
  // for(i=0; i <NTRU_N; i++)  
  r[bid*NTRU_N +tid] = mod3(r[bid*NTRU_N +tid] + 2*r[bid*NTRU_N +NTRU_N-1]);
}

__global__ void poly_S3_tobytes_gpu(uint8_t msg[NTRU_OWCPA_MSGBYTES], uint16_t *a)
{
  int tid = threadIdx.x, i, j, bid = blockIdx.x;
  unsigned char c;

    c =        a[bid*NTRU_N + 5*tid+4] & 255;
    c = (3*c + a[bid*NTRU_N + 5*tid+3]) & 255;
    c = (3*c + a[bid*NTRU_N + 5*tid+2]) & 255;
    c = (3*c + a[bid*NTRU_N + 5*tid+1]) & 255;
    c = (3*c + a[bid*NTRU_N + 5*tid+0]) & 255;
    msg[bid*NTRU_OWCPA_MSGBYTES + tid] = c;

#if NTRU_PACK_DEG > (NTRU_PACK_DEG / 5) * 5  // if 5 does not divide NTRU_N-1

  if(tid==0)
  {
    i = NTRU_PACK_DEG/5;
    c = 0;
    for(j = NTRU_PACK_DEG - (5*i) - 1; j>=0; j--)
      c = (3*c + a[bid*NTRU_N + 5*i+j]) & 255;
    msg[bid*NTRU_OWCPA_MSGBYTES + i] = c;
  }
#endif
}


__global__ void poly_Sq_frombytes_gpu_global(uint16_t *r, const unsigned char *a)
{
  int i = NTRU_PACK_DEG/8;
  uint32_t tid = threadIdx.x, bid = blockIdx.x;
  uint32_t b_offset = bid*NTRU_N, a_offset = bid*NTRU_SECRETKEYBYTES;

    r[b_offset + 8*tid+0] = (a[a_offset + 11*tid+ 0] >> 0) | (((uint16_t)a[a_offset + 11*tid+ 1] & 0x07) << 8);
    r[b_offset + 8*tid+1] = (a[a_offset + 11*tid+ 1] >> 3) | (((uint16_t)a[a_offset + 11*tid+ 2] & 0x3f) << 5);
    r[b_offset + 8*tid+2] = (a[a_offset + 11*tid+ 2] >> 6) | (((uint16_t)a[a_offset + 11*tid+ 3] & 0xff) << 2) | (((uint16_t)a[a_offset + 11*tid+ 4] & 0x01) << 10);
    r[b_offset + 8*tid+3] = (a[a_offset + 11*tid+ 4] >> 1) | (((uint16_t)a[a_offset + 11*tid+ 5] & 0x0f) << 7);
    r[b_offset + 8*tid+4] = (a[a_offset + 11*tid+ 5] >> 4) | (((uint16_t)a[a_offset + 11*tid+ 6] & 0x7f) << 4);
    r[b_offset + 8*tid+5] = (a[a_offset + 11*tid+ 6] >> 7) | (((uint16_t)a[a_offset + 11*tid+ 7] & 0xff) << 1) | (((uint16_t)a[a_offset + 11*tid+ 8] & 0x03) <<  9);
    r[b_offset + 8*tid+6] = (a[a_offset + 11*tid+ 8] >> 2) | (((uint16_t)a[a_offset + 11*tid+ 9] & 0x1f) << 6);
    r[b_offset + 8*tid+7] = (a[a_offset + 11*tid+ 9] >> 5) | (((uint16_t)a[a_offset + 11*tid+10] & 0xff) << 3);
  
  if(tid==0)  //wklee, can be parallelized later
  {
    switch(NTRU_PACK_DEG&0x07)
    {
      // cases 0 and 6 are impossible since 2 generates (Z/n)* and
      // p mod 8 in {1, 7} implies that 2 is a quadratic residue.
      case 4:
        r[b_offset + 8*i+0] = (a[a_offset + 11*i+ 0] >> 0) | (((uint16_t)a[a_offset + 11*i+ 1] & 0x07) << 8);
        r[b_offset + 8*i+1] = (a[a_offset + 11*i+ 1] >> 3) | (((uint16_t)a[a_offset + 11*i+ 2] & 0x3f) << 5);
        r[b_offset + 8*i+2] = (a[a_offset + 11*i+ 2] >> 6) | (((uint16_t)a[a_offset + 11*i+ 3] & 0xff) << 2) | (((uint16_t)a[a_offset + 11*i+ 4] & 0x01) << 10);
        r[b_offset + 8*i+3] = (a[a_offset + 11*i+ 4] >> 1) | (((uint16_t)a[a_offset + 11*i+ 5] & 0x0f) << 7);
        break;
      case 2:
        r[b_offset + 8*i+0] = (a[a_offset + 11*i+ 0] >> 0) | (((uint16_t)a[a_offset + 11*i+ 1] & 0x07) << 8);
        r[b_offset + 8*i+1] = (a[a_offset + 11*i+ 1] >> 3) | (((uint16_t)a[a_offset + 11*i+ 2] & 0x3f) << 5);
        break;
    }
    r[NTRU_N-1] = 0;
  }
}