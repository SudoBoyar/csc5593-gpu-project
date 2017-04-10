CC=g++
BINARIES=jacobi.out

all: $(BINARIES)

jacobi.out: jacobi.cpp data_utils.h parse_args.h sequential.h sequential_blocked.h sequential_overlapped_tiling.h
	$(CC) jacobi.cpp data_utils.h parse_args.h sequential.h sequential_blocked.h sequential_overlapped_tiling.h -o jacobi.out

clean:
	rm -f $(BINARIES)
