CUDA_ARCH ?= sm_35

test: test.cu labels.o centroids.h kmeans.h
	nvcc -arch=$(CUDA_ARCH) -Xptxas -v -o test test.cu timer.o labels.o -lcublas

labels.o: labels.cu labels.h
	nvcc -arch=$(CUDA_ARCH) -Xptxas -v -c -o labels.o labels.cu