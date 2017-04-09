#include "parse_args.h"
#include "matrix_utils.h"
#include "sequential.h"
#include "sequential_blocked.h"

void jacobi_naive(Matrix A, int iterations) {
    Matrix B = copy_matrix(A);
}

int main(int argc, char *argv[]) {
    Args args = parse_arguments(argc, argv);
    Matrix A;
    A = initialize_matrix(args.dimensions, x, y, z);

    //if (args.debug) { print_data(data, args.size, args.dimensions); }
    if (args.blocked) {
    } else {
        jacobi_naive(A, args.iterations);
    }
    if (args.debug) { print_data(data, args.size, args.dimensions); }
}
