# TensorCrypto
This is the code accompanying the paper "TensorCrypto". 

# Introduction
The tensor cores in NVIDIA GPU are exploited to perform polynomial convolution/matrix multiplication found in several lattice-based cryptosystems. In this paper, we have explored TensorTRU (NTRU), TensorLAC (LAC) and TensorFro (Frodo). It can benefit other similar lattice-based schemes that cannot be accelerated by NTT. This repository contains source codes for implementing polynomial convolution in NTRUHPS2048509 and NTRUHPS2048677.

# How to use
There is a Makefile accompanied with the source codes. You can build the executable by typing "make".

Note that you need to change the sm version in GPU to suit your device. The default is -arch=sm_75, suitable for RTX2060,RTX2070 and RTX2080.

1) The flag NTRUHPS2048509 and NTRUHPS2048677 are used to select the respective NTRU parameter sets.
2) The flag K determines how many times we reuse the same public/private key pair. This allows flexible configuration to cater for ephemeral key pair usage.
3) The flag WMMA_THREAD determines how many threads per block for the tensor core. It can be set as any multiple of 32.
