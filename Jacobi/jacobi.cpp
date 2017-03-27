//
// Created by Alex on 3/25/2017.
//
#include <cstdlib>
#include <ctype.h>
#include <errno.h>
#include <iostream>
#include <math.h>
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
int size = 1024, dimensions = 2;
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


    return true;
}

// Sequential implementations

void jacobi1d_sequential(float *data) {
    for (int t = 0; t < iterations; t++) {
        for (int i = 1; i < size - 1; i++) {
            // Include Self
            //data[i] = (data[i] + data[i - 1] + data[i + 1]) / 3;
            // Exclude Self
            data[i] = (data[i - 1] + data[i + 1]) / 2;
        }
    }
}

void jacobi2d_sequential(float *data) {
    for (int t = 0; t < iterations; t++) {
        for (int i = 0; i < size; i++) {
            for (int j = 1; j < size - 1; j++) {
                // Include Self
                //data[i * size + j] = (data[i * size + j] + data[(i - 1) * size + j] + data[(i + 1) * size + j] + data[i * size + j - 1] + data[i * size + j + 1]) / 5;
                // Exclude Self
                data[i * size + j] = (data[(i - 1) * size + j] + data[(i + 1) * size + j] + data[i * size + j - 1] +
                                      data[i * size + j + 1]) / 4;
            }
        }
    }
}

void jacobi3d_sequential(float *data) {
    for (int t = 0; t < iterations; t++) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < size; k++) {
                    // Include Self
                    //data[i * size * size + j * size + k] =
                    //        (
                    //                data[i * size * size + j * size + k] +
                    //                data[(i - 1) * size * size + j * size + k] +
                    //                data[(i + 1) * size * size + j * size + k] +
                    //                data[i * size * size + (j - 1) * size + k] +
                    //                data[i * size * size + (j + 1) * size + k] +
                    //                data[i * size * size + j * size + k - 1] +
                    //                data[i * size * size + j * size + k + 1]
                    //        ) / 7;
                    // Exclude Self
                    data[i * size * size + j * size + k] =
                            (
                                    data[(i - 1) * size * size + j * size + k] +
                                    data[(i + 1) * size * size + j * size + k] +
                                    data[i * size * size + (j - 1) * size + k] +
                                    data[i * size * size + (j + 1) * size + k] +
                                    data[i * size * size + j * size + k - 1] +
                                    data[i * size * size + j * size + k + 1]
                            ) / 6;
                }
            }
        }
    }
}

void jacobi_sequential(float *data) {
    switch (dimensions) {
        case 1:
            jacobi1d_sequential(data);
            break;
        case 2:
            jacobi2d_sequential(data);
            break;
        case 3:
            jacobi3d_sequential(data);
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
    for (int i = 0; i < size; i++) {
        // Top Row
        //data[i] = 1.0;
        // Bottom Row
        //data[(size - 1) * size + i] = 1.0;
        // Left Column
        data[i * size] = 1.0;
        // Right Column
        data[i * size + size - 1] = 1.0;
    }
}

void initialize_data_3d(float *data) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            // Face 1: X = 0 Plane
            data[i * size + j] = 1.0;
            // Face 2: X = N Plane
            data[(size - 1) * size * size + i * size + j] = 1.0;
            // Face 3: Y = 0 Plane
            //data[i * size * size + j] = 1.0;
            // Face 4: Y = N Plane
            //data[i * size * size + (size - 1) * size + j] = 1.0;
            // Face 5: Z = 0 Plane
            //data[i * size * size + j * size] = 1.0;
            // Face 6: Z = N Plane
            //data[i * size * size + j * size + size - 1] = 1.0;
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

void print_matrix(float *data) {
    if (size > 10) {
        cerr << "Matrix too big to print\n" << endl;
        return;
    }

    if (dimensions == 1) {
        for (int i = 0; i < size; i++) {
            fprintf(stdout, "%.3f ", data[i]);
        }
    } else if (dimensions == 2) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                fprintf(stdout, "%.3f ", data[i * size + j]);
            }
            cout << endl;
        }
    } else if (dimensions == 3) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < size; k++) {
                    fprintf(stdout, "%.3f ", data[i * size * size + j * size + k]);
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
    float *data = new float[(int)pow(size, dimensions) * 2];

    initialize_data(data);
    if (debug) { print_matrix(data); }
    if (sequential) {
        jacobi_sequential(data);
    } else {
        // Add CUDA calls
    }
    if (debug) { print_matrix(data); }
}