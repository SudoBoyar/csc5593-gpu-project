#include "parse_args.h"
#include "matrix_utils.h"
#include "naive.cu"

// TODO Parameterize dims.
void jacobi_naive(Args args, Matrix A, Matrix B) {
    if (args.dimensions == 1) {
        //dim3 blocks(args.grid_size);
        //dim3 threads(args.size/args.grid_size);
        dim3 blocks(args.size/32);
        dim3 threads(32);

        for (int t = 0; t < args.iterations; t++) {
            jacobi1d_naive<<<blocks, threads>>>(A, B);
        }
    } else if (args.dimensions == 2) {
        dim3 blocks(args.size/16, args.size/16);
        dim3 threads(16, 16);
        for (int t = 0; t < args.iterations; t++) {
            jacobi2d_naive<<<blocks, threads>>>(A, B);
        }
    } else {
        dim3 blocks(args.size/8, args.size/8, args.size/8);
        dim3 threads(8, 8, 8);
        for (int t = 0; t < args.iterations; t++) {
            jacobi3d_naive<<<blocks, threads>>>(A, B);
        }
    }
}

int main(int argc, char *argv[]) {
    Args args = parse_arguments(argc, argv);
    Matrix A, B;
    A = initialize_matrix(args.dimensions, x, y, z);
    B = initialize_matrix(args.dimensions, x, y, z);

    //if (args.debug) { print_data(data, args.size, args.dimensions); }
    if (args.blocked) {
    } else {
        jacobi_naive(args, A, B);
    }
    if (args.debug) { print_data(data, args.size, args.dimensions); }
}
