all: simpleTensorCoreGEMM.cu
	nvcc -o TCGemm -arch=sm_75 simpleTensorCoreGEMM.cu

clean:
	rm -f TCGemm


