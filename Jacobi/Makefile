CC=g++
BINARIES=jacobi

all: $(BINARIES)

jacobi: jacobi.cpp data_utils.h parse_args.h sequential.h sequential_blocked.h
	$(CC) jacobi.cpp data_utils.h parse_args.h sequential.h sequential_blocked.h -o jacobi

clean:
	rm -f $(BINARIES)
