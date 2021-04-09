#define NTRU_N_PWR2 512	// Multiple of 32
#define PADDING 3		// 512 - 3 = 509
#define K 496			// Multiple of 16
#define BATCH K
#define DEBUG 		// To check the encryption/decryption results

#define MODQ(X) ((X) & (NTRU_Q-1))


void ntru_enc(uint8_t mode, uint16_t *h_m, uint16_t *h_r, uint8_t *h_pk, uint8_t *h_c) ;
void ntru_dec(uint8_t mode, uint8_t *h_rm, uint8_t *h_c, uint8_t *h_sk, uint16_t *h_m) ;


