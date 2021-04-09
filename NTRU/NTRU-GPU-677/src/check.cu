#include "../include/check.cuh"

#include <stdlib.h> 

__global__ void owcpa_check_ciphertext_gpu(int *fail, uint8_t *ciphertext)
{
  /* A ciphertext is log2(q)*(n-1) bits packed into bytes.  */
  /* Check that any unused bits of the final byte are zero. */

  uint16_t t = 0;
  uint32_t tid = threadIdx.x, bid = blockIdx.x;

  t = ciphertext[tid*NTRU_CIPHERTEXTBYTES + NTRU_CIPHERTEXTBYTES-1];
  t &= 0xff << (8-(7 & (NTRU_LOGQ*NTRU_PACK_DEG)));

  /* We have 0 <= t < 256 */
  /* Return 0 on success (t=0), 1 on failure */
  // return (int) (1&((~t + 1) >> 15));
  fail[bid] |= (int) (1&((~t + 1) >> 15));  
}

__global__ void owcpa_check_r_gpu(int *fail, uint16_t *r)
{
  /* A valid r has coefficients in {0,1,q-1} and has r[N-1] = 0 */
  /* Note: We may assume that 0 <= r[i] <= q-1 for all i        */

  int i;
  uint32_t t = 0;
  uint32_t bid = blockIdx.x;
  uint16_t c;
  for(i=0; i<NTRU_N-1; i++)
  {
    c = r[bid*NTRU_N + i];
    t |= (c + 1) & (NTRU_Q-4);  /* 0 iff c is in {-1,0,1,2} */
    t |= (c + 2) & 4;  /* 1 if c = 2, 0 if c is in {-1,0,1} */
  }
  t |= r[NTRU_N-1]; /* Coefficient n-1 must be zero */

  /* We have 0 <= t < 2^16. */
  /* Return 0 on success (t=0), 1 on failure */
  // return (int) (1&((~t + 1) >> 31));
  fail[bid] |= (int) (1&((~t + 1) >> 31));  
}

__global__ void owcpa_check_m_gpu(int *fail, uint16_t *m)
{
  /* Check that m is in message space, i.e.                  */
  /*  (1)  |{i : m[i] = 1}| = |{i : m[i] = 2}|, and          */
  /*  (2)  |{i : m[i] != 0}| = NTRU_WEIGHT.                  */
  /* Note: We may assume that m has coefficients in {0,1,2}. */
    uint32_t bid = blockIdx.x;
    int i;
    uint32_t t = 0;
    uint16_t ps = 0;
    uint16_t ms = 0;
    for(i=0; i<NTRU_N; i++)
    {
        ps += m[bid*NTRU_N + i] & 1;
        ms += m[bid*NTRU_N + i] & 2;
    }
    t |= ps ^ (ms >> 1);   /* 0 if (1) holds */
    t |= ms ^ NTRU_WEIGHT; /* 0 if (1) and (2) hold */

      /* We have 0 <= t < 2^16. */
      /* Return 0 on success (t=0), 1 on failure */
      // return (int) (1&((~t + 1) >> 31));
    fail[bid] |= (int) (1&((~t + 1) >> 31));    
}
