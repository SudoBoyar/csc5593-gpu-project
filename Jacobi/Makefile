CC=g++
BINARIES=jacobi_seq naive_gpu blocked_gpu overlapped_gpu cps_gpu cpssm_gpu

TARGET_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

all: $(BINARIES)

jacobi_seq: jacobi.cpp data_utils.h parse_args.h sequential.h sequential_blocked.h sequential_overlapped_tiling.h
	$(CC) -std=c++11 jacobi.cpp data_utils.h parse_args.h sequential.h sequential_blocked.h sequential_overlapped_tiling.h -o jacobi_seq

naive_gpu: $(TARGET_DIR)/single_file_naive.cu
	ssh node18 nvcc -std=c++11 $(TARGET_DIR)/single_file_naive.cu -o $(TARGET_DIR)/naive_gpu

blocked_gpu: $(TARGET_DIR)/single_file_blocked.cu
	ssh node18 nvcc -std=c++11 $(TARGET_DIR)/single_file_blocked.cu -o $(TARGET_DIR)/blocked_gpu

overlapped_gpu: $(TARGET_DIR)/single_file_overlapped.cu
	ssh node18 nvcc -std=c++11 $(TARGET_DIR)/single_file_overlapped.cu -o $(TARGET_DIR)/overlapped_gpu

cps_gpu: $(TARGET_DIR)/cached_plane.cu
	ssh node18 nvcc -std=c++11 $(TARGET_DIR)/cached_plane.cu -o $(TARGET_DIR)/cps_gpu

cpssm_gpu: $(TARGET_DIR)/cached_plane_shared_mem.cu
	ssh node18 nvcc -std=c++11 $(TARGET_DIR)/cached_plane_shared_mem.cu -o $(TARGET_DIR)/cpssm_gpu

clean:
	rm -f $(BINARIES)
