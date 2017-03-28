//
// Created by Alex on 3/25/2017.
//

#include "parse_args.h"
#include "data_utils.h"
#include "sequential.h"

using namespace std;


// Main

int main(int argc, char *argv[]) {
    Args args = parse_arguments(argc, argv);
    float *data = new float[args.alloc_size];
    float *temp = new float[args.alloc_size];

    initialize_data(data, args.size, args.dimensions);
    initialize_data(temp, args.size, args.dimensions);

    if (args.debug) { print_data(data, args.size, args.dimensions); }
    if (args.sequential) {
        jacobi_sequential(data, temp, args.iterations, args.size, args.dimensions);
    } else {
        // Add CUDA calls
    }
    if (args.debug) { print_data(data, args.size, args.dimensions); }
}