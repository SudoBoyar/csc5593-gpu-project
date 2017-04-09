//
// Created by Alex on 3/25/2017.
//

#include "parse_args.h"
#include "data_utils.h"
#include "sequential.h"
#include "sequential_blocked.h"
#include "sequential_overlapped_tiling.h"

using namespace std;


// Main

int main(int argc, char *argv[]) {
    Args args = parse_arguments(argc, argv);
    float *data = new float[args.alloc_size];

    initialize_data(data, args.size, args.dimensions);

    //if (args.debug) { print_data(data, args.size, args.dimensions); }
    if (args.blocked) {
        float *temp = new float[args.alloc_size];
        initialize_data(temp, args.size, args.dimensions);

        jacobi_sequential_blocked(data, temp, args.iterations, args.size, args.dimensions, args.xBlockSize,
                                  args.yBlockSize, args.zBlockSize);
    } else if (args.overlapped) {
        float *temp = new float[args.alloc_size * (args.tBlockSize - 1)];
        for (int i = 0; i < args.tBlockSize; i++) {
            initialize_data(temp + i * args.alloc_size, args.size, args.dimensions);
        }

        jacobi_sequential_overlapped(data, temp, args.iterations, args.size, args.dimensions, args.tBlockSize,
                                     args.xBlockSize, args.yBlockSize, args.zBlockSize);
    } else {
        float *temp = new float[args.alloc_size];
        initialize_data(temp, args.size, args.dimensions);

        jacobi_sequential(data, temp, args.iterations, args.size, args.dimensions);
    }
    if (args.debug) { print_data(data, args.size, args.dimensions); }
}