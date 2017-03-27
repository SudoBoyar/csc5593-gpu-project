//
// Created by Alex on 3/25/2017.
//
#include <cstdlib>
#include <ctype.h>
#include <errno.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

// Shorthand for formatting usage options
#define fpe(msg) fprintf(stderr, "\t%s\n", msg);

using namespace std;

bool debug = false;
bool sequential = false;

// Data attributes
int size = 1024, dimensions = 2, alloc_size;
// Run attributes
int gridSize = 1, blockSize = 16, iterations = 1000;

static void usage(char *prog_name, string msg) {
    if (msg.size() > 0) {
        fputs(msg.c_str(), stderr);
    }

    fprintf(stderr, "Usage: %s [options]\n", prog_name);
    fprintf(stderr, "Options are:\n");
    fpe("-n<size> Set data size (default: 1024)");
    fpe("-d<size> Set number of data dimensions (1, 2, or 3) (default: 2)");
    fpe("-g<size> Set grid size (default: 1)");
    fpe("-b<size> Set block size (default: 16)");
    fpe("-i<iter> Number of iterations to perform (default: 1000)");
    fpe("-S       Execute sequential, CPU version");
    fpe("-D       Print debug info");
    fpe("-h       Print usage info (this message)");
    exit(EXIT_FAILURE);
}

bool parse_arguments(int argc, char *argv[]) {
    int opt;
    // Parse args
    while ((opt = getopt(argc, argv, "n:d:g:b:i:hSD")) != -1) {
        switch (opt) {
            case 'D':
                debug = true;
                break;
            case 'S':
                sequential = true;
                break;
            case 'n':
                size = atoi(optarg);
                break;
            case 'd':
                dimensions = atoi(optarg);
                break;
            case 'g':
                gridSize = atoi(optarg);
                break;
            case 'b':
                blockSize = atoi(optarg);
                break;
            case 'i':
                iterations = atoi(optarg);
                break;
            case 'h':
                usage(argv[0], "");
                break;
            default:
                usage(argv[0], "Unrecognized option\n");
        }
    }

    // check sizes
    if (size <= 0) {
        cout << "Data size must be larger than 0" << endl;
        exit(EXIT_FAILURE);
    }

    if (dimensions <= 0 || dimensions >= 4) {
        cerr << "Data must be 1, 2, or 3 dimensions" << endl;
        exit(EXIT_FAILURE);
    }

    if (dimensions == 1) {
        alloc_size = size;
    } else if (dimensions == 2) {
        alloc_size = size * size;
    } else {
        alloc_size = size * size * size;
    }

    return true;
}

// Sequential implementations

void jacobi1d_sequential(float *data, float *temp) {
    float *swap;
    for (int t = 0; t < iterations; t++) {
        for (int x = 1; x < size - 1; x++) {
            // Include Self
            temp[x] = (data[x] + data[x - 1] + data[x + 1]) / 3;
            // Exclude Self
            //temp[x] = (data[x - 1] + data[x + 1]) / 2;
        }

        swap = data;
        data = temp;
        temp = swap;
    }
}

void jacobi2d_sequential(float *data, float *temp) {
    float *swap;
    for (int t = 0; t < iterations; t++) {
        for (int y = 1; y < size - 1; y++) {
            for (int x = 1; x < size - 1; x++) {
                // Include Self
                temp[y * size + x] =
                    (
                        data[y * size + x] +
                        data[(y - 1) * size + x] +
                        data[(y + 1) * size + x] +
                        data[y * size + x - 1] +
                        data[y * size + x + 1]
                    ) / 5;
                // Exclude Self
                //temp[y * size + x] =
                //    (
                //        data[(y - 1) * size + x] +
                //        data[(y + 1) * size + x] +
                //        data[y * size + x - 1] +
                //        data[y * size + x + 1]
                //    ) / 4;
            }
        }

        swap = data;
        data = temp;
        temp = swap;
    }
}

void jacobi3d_sequential(float *data, float *temp) {
    float *swap;
    for (int t = 0; t < iterations; t++) {
        for (int z = 1; z < size - 1; z++) {
            for (int y = 1; y < size - 1; y++) {
                for (int x = 1; x < size - 1; x++) {
                    // Include Self
                    temp[z * size * size + y * size + x] =
                        (
                            data[z * size * size + y * size + x] +
                            data[(z - 1) * size * size + y * size + x] +
                            data[(z + 1) * size * size + y * size + x] +
                            data[z * size * size + (y - 1) * size + x] +
                            data[z * size * size + (y + 1) * size + x] +
                            data[z * size * size + y * size + x - 1] +
                            data[z * size * size + y * size + x + 1]
                        ) / 7;
                    // Exclude Self
                    //temp[z * size * size + y * size + x] =
                    //    (
                    //        data[(z - 1) * size * size + y * size + x] +
                    //        data[(z + 1) * size * size + y * size + x] +
                    //        data[z * size * size + (y - 1) * size + x] +
                    //        data[z * size * size + (y + 1) * size + x] +
                    //        data[z * size * size + y * size + x - 1] +
                    //        data[z * size * size + y * size + x + 1]
                    //    ) / 6;
                }
            }
        }

        swap = data;
        data = temp;
        temp = swap;
    }
}

void jacobi_sequential(float *data, float *temp) {
    switch (dimensions) {
        case 1:
            jacobi1d_sequential(data, temp);
            break;
        case 2:
            jacobi2d_sequential(data, temp);
            break;
        case 3:
            jacobi3d_sequential(data, temp);
            break;
    }
}

// Data Initialization

void initialize_data_1d(float *data) {
    // Left
    data[0] = 1.0;
    // Right
    data[size - 1] = 1.0;
}

void initialize_data_2d(float *data) {
    int x, y;

    for (int i = 0; i < size; i++) {
        // Top Row
        x = i;
        y = 0;
        data[y * size + x] = 1.0;
        // Bottom Row
        x = i;
        y = size - 1;
        data[y * size + x] = 1.0;
        // Left Column
        x = 0;
        y = i;
        data[y * size + x] = 1.0;
        // Right Column
        x = size - 1;
        y = i;
        data[y * size + x] = 1.0;
    }
}

void initialize_data_3d(float *data) {
    int x, y, z;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            // Face 1: X = 0 Plane
            x = 0;
            y = j;
            z = i;
            data[z * size * size + y * size + x] = 1.0;
            // Face 2: X = N Plane
            x = size - 1;
            y = j;
            z = i;
            data[z * size * size + y * size + x] = 1.0;
            // Face 3: Y = 0 Plane
            x = j;
            y = 0;
            z = i;
            data[z * size * size + y * size + x] = 1.0;
            // Face 4: Y = N Plane
            x = j;
            y = size - 1;
            z = i;
            data[z * size * size + y * size + x] = 1.0;
            // Face 5: Z = 0 Plane
            x = j;
            y = i;
            z = 0;
            data[z * size * size + y * size + x] = 1.0;
            // Face 6: Z = N Plane
            x = j;
            y = i;
            z = size - 1;
            data[z * size * size + y * size + x] = 1.0;
        }
    }
}

void initialize_data(float *data) {
    switch (dimensions) {
        case 1:
            initialize_data_1d(data);
            break;
        case 2:
            initialize_data_2d(data);
            break;
        case 3:
            initialize_data_3d(data);
            break;
    }
}


// Debug Helpers

void print_data(float *data) {
    if (size > 13) {
        cerr << "Data too big to print\n" << endl;
        return;
    }

    if (dimensions == 1) {
        for (int x = 0; x < size; x++) {
            fprintf(stdout, "%.3f ", data[x]);
        }
    } else if (dimensions == 2) {
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                fprintf(stdout, "%.3f ", data[y * size + x]);
            }
            cout << endl;
        }
    } else if (dimensions == 3) {
        for (int z = 0; z < size; z++) {
            for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                    fprintf(stdout, "%.3f ", data[z * size * size + y * size + x]);
                }
                cout << endl;
            }
            cout << endl;
        }
    }
    cout << endl << endl;
}


// Main

int main(int argc, char *argv[]) {
    parse_arguments(argc, argv);
    float *data = new float[alloc_size];
    float *temp = new float[alloc_size];

    initialize_data(data);
    initialize_data(temp);

    if (debug) { print_data(data); }
    if (sequential) {
        jacobi_sequential(data, temp);
    } else {
        // Add CUDA calls
    }
    if (debug) { print_data(data); }
}