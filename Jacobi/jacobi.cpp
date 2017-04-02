//
// Created by Alex on 3/25/2017.
//

#include "parse_args.h"
#include "data_utils.h"
#include "sequential.h"
#include "sequential_blocked.h"

using namespace std;


// Main

int main(int argc, char *argv[]) {
    Args args = parse_arguments(argc, argv);
    float *data = new float[args.alloc_size];
    float *temp = new float[args.alloc_size];

    initialize_data(data, args.size, args.dimensions);
    initialize_data(temp, args.size, args.dimensions);

    //if (args.debug) { print_data(data, args.size, args.dimensions); }
    if (args.sequential) {
        if (args.blocked) {
            jacobi_sequential_blocked(data, temp, args.iterations, args.size, args.dimensions, args.xBlockSize,
                                      args.yBlockSize, args.zBlockSize);
        } else {
            jacobi_sequential(data, temp, args.iterations, args.size, args.dimensions);
        }
    } else {
        // Add CUDA calls
    }
    if (args.debug) { print_data(data, args.size, args.dimensions); }
}