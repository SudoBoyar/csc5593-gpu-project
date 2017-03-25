//
// Created by Alex on 3/25/2017.
//
#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

// Shorthand for formatting usage options
#define fpe(msg) fprintf(stderr, "\t%s\n", msg);

bool debug = false;
bool sequential = false;

// Data attributes
int size = 1024, dimensions = 2;
// Run attributes
int gridSize = 1, blockSize = 16, iterations = 1000;

virtual struct Matrix {
    virtual float adjacent_sum();

    virtual int adjacent_count();
};

struct Matrix1D : Matrix {
    float *data = new float[size];

    void set(i, value) {
        data[i] = value;
    }

    float get(i) {
        return data[i];
    }

    float self(i) {
        return get(i);
    }

    virtual float adjacent_sum(i) {
        return left(i) + right(i);
    }

    virtual int adjacent_count(i) {
        return 2;
    }

    float left(i) {
        return get(i - 1);
    }

    float right(i) {
        return get(i + 1);
    }
};

struct Matrix2D {
    float *data = new float[size * size];

    void set(i, j, value) {
        data[i * size + j] = value;
    }

    float get(i, j) {
        return data[i * size + j];
    }

    float self(i, j) {
        return get(i, j);
    }

    virtual float adjacent_sum(i, j) {
        return left(i, j) + right(i, j) + up(i, j) + down(i, j);
    }

    virtual int adjacent_count(i) {
        return 4;
    }

    float left(i, j) {
        return get(i, j - 1);
    }

    float right(i, j) {
        return get(i, j + 1);
    }

    float up(i, j) {
        return get(i - 1, j);
    }

    float down(i, j) {
        return get(i + 1, j);
    }
};

struct Matrix3D {
    float *data = new float[size * size * size];

    void set(i, j, k, value) {
        data[i * size * size + j * size + k] = value;
    }

    float get(i, j, k) {
        return data[i * size * size + j * size + k];
    }

    float self(i, j, k) {
        return get(i, j, k);
    }

    virtual float adjacent_sum(i, j, k) {
        return left(i, j, k) + right(i, j, k) + up(i, j, k) + down(i, j, k) + away(i, j, k) + toward(i, j, k);
    }

    virtual int adjacent_count(i) {
        return 6;
    }

    float up(i, j, k) {
        return get(i - 1, j, k);
    }

    float down(i, j, k) {
        return get(i + 1, j, k);
    }

    float left(i, j, k) {
        return get(i, j - 1, k);
    }

    float right(i, j, k) {
        return get(i, j + 1, k);
    }

    float away(i, j, k) {
        return get(i, j, k - 1);
    }

    float toward(i, j, k) {
        return get(i, j, k + 1);
    }
};

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
            case 'n':
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
        cerr << "Data size must be larger than 0" << endl;
        exit(EXIT_FAILURE);
    }

    if (dimensions <= 0 || dimensions >= 4) {
        cerr << "Data must be 1, 2, or 3 dimensions" << endl;
        exit(EXIT_FAILURE);
    }

    return true;
}

// Sequential implementations
void jacobi_sequential(Matrix1D *data) {
    for (int t = 0; t < iterations; t++) {
        for (i = 1; i < size - 1; i++) {
            data->set(i,
                      (
                          data->self(i) +
                          data->left(i) +
                          data->right(i)
                      ) / 3);
        }
    }
}

void jacobi_sequential(Matrix2D *data) {
    for (int t = 0; t < iterations; t++) {
        for (int i = 1; i < size - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                data->set(i, j,
                          (
                              data->self(i, j) +
                              data->left(i, j) +
                              data->right(i, j) +
                              data->up(i, j) +
                              data->down(i, j)
                          ) / 5
                );
            }
        }
    }
}

void jacobi_sequential(Matrix3D *data) {
    for (int t = 0; t < iterations; t++) {
        for (int i = 1; i < size - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                for (int k = 1; k < size - 1; k++) {
                    data->set(i, j, k,
                              (
                                  data->self(i, j, k) +
                                  data->up(i, j, k) +
                                  data->down(i, j, k) +
                                  data->left(i, j, k) +
                                  data->right(i, j, k) +
                                  data->away(i, j, k) +
                                  data->toward(i, j, k)
                              ) / 7
                    );
                }
            }
        }
    }
}

void initialize_data(Matrix1D *data) {
    data->set(0, 1.0);
    data->set(size - 1, 1.0);
}

void initialize_data(Matrix2D *data) {
    for (int i = 0; i < size; i++) {
        data.set(i, 0, 1.0);
        data.set(i, size - 1, 1.0);
        data.set(0, i, 1.0);
        data.set(size - 1, i, 1.0);
    }
}

void initialize_data(Matrix3D *data) {
    for (int i = 0; i < size; i++) {
        data.set(i, 0, 0, 1.0);
        data.set(i, size - 1, 0, 1.0);
        data.set(0, i, 1.0);
        data.set(size - 1, i, 1.0);
    }
}


int main(int argc, char *argv[]) {
    parse_arguments(argc, argv);

    switch (dimensions) {
        case 1:
            Matrix1D data = new Matrix1D();
            break;
        case 2:
            Matrix2D data = new Matrix2D();
            break;
        case 3:
            Matrix3D data = new Matrix3D();
            break;
    }

    initialize_data(&data);

    if (sequential) {
        jacobi_sequential(&data);
    } else {
        // Add CUDA calls
    }
}