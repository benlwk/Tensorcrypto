# TensorCrypto
This is the code accompanying the paper "TensorCrypto". 
https://eprint.iacr.org/2021/173

# Introduction
The tensor cores in NVIDIA GPU are exploited to perform polynomial convolution/matrix multiplication found in several lattice-based cryptosystems. In this paper, we have explored TensorTRU (NTRU), TensorLAC (LAC) and TensorFro (Frodo). It can benefit other similar lattice-based schemes that cannot be accelerated by NTT. This repository contains source codes for implementing polynomial convolution in NTRUHPS2048509 and NTRUHPS2048677. The technique can be extended to implement LAC and variants of FrodoKEM as well.

# How to use
There is a Makefile accompanied with the source codes in each separate folder. You can build the executable by typing "make".

Note that you need to change the sm version in GPU to suit your device. The default is -arch=sm_75, which is suitable for RTX2060,RTX2070 and RTX2080.

1) The NTRU public-key encryption (see NTRU-GPU-509 and NTRU-GPU-677) is implemented using ordinary GPU cores and tensor cores. You can select them by passing 0 or 1 to the executable. 

For instance, this executes NTRU with ordinary GPU cores: 
$ ./runtest -m 0 
This executes NTRU with ordinary tensor cores: 
$ ./runtest -m 1

3) The folder TensorTRU contains only the polynomial convolution with NTRU parameter sets. The flag NTRUHPS2048509 and NTRUHPS2048677 are used to select the respective NTRU parameter sets. You may also comment out the ENCRYPT flag for NTRU decryption. The flag K determines how many times we reuse the same public/private key pair. This allows a flexible configuration to cater for ephemeral key pair usage. The flag WMMA_THREAD determines how many threads per block for the tensor core. It can be set as any multiple of 32.
4) Similarly, the folders TensorLAC and TensorFro contains the code for executing polynomial convolution with different LAC and FrodoKEM parameter sets.
