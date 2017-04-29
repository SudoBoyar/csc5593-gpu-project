CC=g++
BINARIES=jacobi_seq naive_gpu blocked_gpu overlapped_gpu

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
JACOBI_PATH := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

all: $(BINARIES)

jacobi_seq: jacobi.cpp data_utils.h parse_args.h sequential.h sequential_blocked.h sequential_overlapped_tiling.h
	$(CC) -std=c++11 jacobi.cpp data_utils.h parse_args.h sequential.h sequential_blocked.h sequential_overlapped_tiling.h -o jacobi_seq

naive_gpu: single_file_naive.cu
	ssh node18 nvcc -std=c++11 $(JACOBI_PATH)/single_file_naive.cu -o $(JACOBI_PATH)/naive_gpu

blocked_gpu: single_file_blocked.cu
	ssh node18 nvcc -std=c++11 $(JACOBI_PATH)/single_file_blocked.cu -o $(JACOBI_PATH)/blocked_gpu

overlapped_gpu: single_file_overlapped.cu
	ssh node18 nvcc -std=c++11 $(JACOBI_PATH)/single_file_overlapped.cu -o $(JACOBI_PATH)/overlapped_gpu

clean:
	rm -f $(BINARIES) jacobi_gpu
