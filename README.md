# TensorCrypto
This is the code accompanying the paper "TensorCrypto". The tensor cores in NVIDIA GPU are exploited to perform polynomial convolution/matrix multiplication found in several lattice-based cryptosystems. In this paper, we have explored TensorTRU (NTRU), TensorLAC (LAC) and TensorFro (Frodo). It can benefit other similar lattice-based schemes that cannot be accelerated by NTT.

usage: there is a Makefile accompanied with the source codes. You can build the executable by typing "make".

Note that you need to change the sm version in GPU to suit your device. The default is -arch=sm_75, suitable for RTX2060,RTX2070 and RTX2080.
