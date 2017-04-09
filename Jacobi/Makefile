CC=g++
BINARIES=jacobi

all: $(BINARIES)

jacobi: jacobi.cpp data_utils.h parse_args.h sequential.h sequential_blocked.h sequential_overlapped_tiling.h
	$(CC) jacobi.cpp data_utils.h parse_args.h sequential.h sequential_blocked.h sequential_overlapped_tiling.h -o jacobi

clean:
	rm -f $(BINARIES)
