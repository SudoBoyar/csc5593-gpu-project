CC=g++
BINARIES=jacobi_seq

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
JACOBI_PATH := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

all: $(BINARIES)

jacobi_seq: jacobi.cpp data_utils.h parse_args.h sequential.h sequential_blocked.h sequential_overlapped_tiling.h
	$(CC) jacobi.cpp data_utils.h parse_args.h sequential.h sequential_blocked.h sequential_overlapped_tiling.h -o jacobi_seq

jacobi_gpu: jacobi.cu matrix.h matrix_utils.h naive.cu
	ssh node18 nvcc $(JACOBI_PATH)/jacobi.cu $(JACOBI_PATH)/matrix.h $(JACOBI_PATH)/matrix_utils.h $(JACOBI_PATH)/naive.cu -o jacobi_gpu

clean:
	rm -f $(BINARIES) jacobi_gpu
