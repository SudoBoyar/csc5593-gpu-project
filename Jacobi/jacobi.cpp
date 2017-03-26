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

struct Point {
    int i, j, k;

    Point(int i = 0, int j = 0, int k = 0) {
        this.i = i;
        this.j = j;
        this.k = k;
    }
};

virtual struct Matrix {
    virtual float adjacent_sum(Point point);
};

struct Matrix1D : Matrix {
    float *data = new float[size];

    void set(Point point, value) {
        data[point.i] = value;
    }

    float get(Point point) {
        return data[point.i];
    }

    float get(int i) {
        return data[i];
    }

    float self(Point point) {
        return get(point.i);
    }

    virtual float adjacent_sum(Point point) {
        return left(point) + right(point);
    }

    float left(Point point) {
        return get(point.i - 1);
    }

    float right(Point point) {
        return get(point.i + 1);
    }
};

struct Matrix2D {
    float *data = new float[size * size];

    void set(Point point, value) {
        data[point.i * size + point.j] = value;
    }

    float get(Point point) {
        return data[point.i * size + point.j];
    }

    float get(int i, int j) {
        return data[i * size + j];
    }

    float self(Point point) {
        return get(point.i, point.j);
    }

    virtual float adjacent_sum(Point point) {
        return left(point) + right(point) + up(point) + down(point);
    }

    float left(Point point) {
        return get(point.i, point.j - 1);
    }

    float right(Point point) {
        return get(point.i, point.j + 1);
    }

    float up(Point point) {
        return get(point.i - 1, point.j);
    }

    float down(Point point) {
        return get(point.i + 1, point.j);
    }
};

struct Matrix3D {
    float *data = new float[size * size * size];

    void set(Point point, value) {
        data[i * size * size + j * size + k] = value;
    }

    float get(Point point) {
        return get(point.i, point.j, point.k);
    }

    float get(int i, int j, int k) {
        return data[i * size * size + j * size + k];
    }

    float self(Point point) {
        return get(point);
    }

    virtual float adjacent_sum(Point point) {
        return left(i, j, k) + right(i, j, k) + up(i, j, k) + down(i, j, k) + away(i, j, k) + toward(i, j, k);
    }

    float up(Point point) {
        return get(point.i - 1, point.j, point.k);
    }

    float down(Point point) {
        return get(point.i + 1, point.j, point.k);
    }

    float left(Point point) {
        return get(point.i, point.j - 1, point.k);
    }

    float right(Point point) {
        return get(point.i, point.j + 1, point.k);
    }

    float away(Point point) {
        return get(point.i, point.j, point.k - 1);
    }

    float toward(Point point) {
        return get(point.i, point.j, point.k + 1);
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
            data->set(Point(i), (data->self(i) + data->adjacent_sum(Point(i))) / 3);
        }
    }
}

void jacobi_sequential(Matrix2D *data) {
    for (int t = 0; t < iterations; t++) {
        for (int i = 1; i < size - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                data->set(Point(i, j), (data->self(Point(i, j)) + data->adjacent_sum(Point(i, j))) / 5);
            }
        }
    }
}

void jacobi_sequential(Matrix3D *data) {
    for (int t = 0; t < iterations; t++) {
        for (int i = 1; i < size - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                for (int k = 1; k < size - 1; k++) {
                    data->set(Point(i, j, k), (data->self(i, j, k) + data->adjacent_sum(Point(i, j, k))) / 7);
                }
            }
        }
    }
}

void jacobi1d_sequential(float *data) {
    for (int t = 0; t < iterations; t++) {
        for (i = 1; i < size - 1; i++) {
            data[i] = (data[i] + data[i - 1] + data[i + 1]) / 3;
        }
    }
}

void jacobi2d_sequential(float *data) {
    for (int t = 0; t < iterations; t++) {
        for (int i = 1; i < size - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                data[i * size + j] =
                    (
                        data[i * size + j] +
                        data[(i - 1) * size + j] +
                        data[(i + 1) * size + j] +
                        data[i * size + j - 1] +
                        data[i * size + j + 1]
                    ) / 5;
            }
        }
    }
}

void jacobi3d_sequential(float *data) {
    for (int t = 0; t < iterations; t++) {
        for (int i = 1; i < size - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                for (int k = 1; k < size - 1; k++) {
                    data[i * size * size + j * size + k] =
                        (
                            data[i * size * size + j * size + k] +
                            data[(i - 1) * size * size + j * size + k] +
                            data[(i + 1) * size * size + j * size + k] +
                            data[i * size * size + (j - 1) * size + k] +
                            data[i * size * size + (j + 1) * size + k] +
                            data[i * size * size + j * size + k - 1] +
                            data[i * size * size + j * size + k + 1]
                        ) / 7;
                }
            }
        }
    }
}

// Data Initialization
void initialize_data(Matrix1D *data) {
    // Left
    data->set(Point(0), 1.0);
    // Right
    data->set(Point(size - 1), 1.0);
}

void initialize_data_1d(float *data) {
    // Left
    data[0] = 1.0;
    // Right
    data[size-1] = 1.0;
}

void initialize_data(Matrix2D *data) {
    for (int i = 0; i < size; i++) {
        // Top Row
        data.set(0, i, 1.0);
        // Bottom Row
        data.set(size - 1, i, 1.0);
        // Left Column
        data.set(Point(i, 0), 1.0);
        // Right Column
        data.set(Point(i, size - 1), 1.0);
    }
}

void initialize_data_2d(float *data) {
    for (int i = 0; i < size; i++) {
        // Top Row
        data[i] = 1.0;
        // Bottom Row
        data[(size - 1) * size + i] = 1.0;
        // Left Column
        data[i * size] = 1.0;
        // Right Column
        data[i * size + size - 1] = 1.0;
    }
}

void initialize_data(Matrix3D *data) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            // Face 1: X = 0 Plane
            data.set(Point(0, i, j), 1.0);
            // Face 2: X = N Plane
            data.set(Point(size - 1, i, j), 1.0);
            // Face 3: Y = 0 Plane
            data.set(Point(i, 0, j), 1.0);
            // Face 4: Y = N Plane
            data.set(Point(i, size - 1, j), 1.0);
            // Face 5: Z = 0 Plane
            data.set(Point(i, j, 0), 1.0);
            // Face 6: Z = N Plane
            data.set(Point(i, j, size - 1), 1.0);
        }
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
            data[i * size * size + j] = 1.0;
            // Face 4: Y = N Plane
            data[i * size * size + (size - 1) * size + j] = 1.0;
            // Face 5: Z = 0 Plane
            data[i * size * size + j * size] = 1.0;
            // Face 6: Z = N Plane
            data[i * size * size + j * size + size - 1] = 1.0;
        }
    }
}


int main(int argc, char *argv[]) {
    parse_arguments(argc, argv);
    Matrix data;
    float* arr_data;

    switch (dimensions) {
        case 1:
            data = new Matrix1D();
            arr_data = new float[size];
            break;
        case 2:
            data = new Matrix2D();
            arr_data = new float[size*size];
            break;
        case 3:
            data = new Matrix3D();
            arr_data = new float[size*size*size];
            break;
    }

    initialize_data(&data);

    if (sequential) {
        jacobi_sequential(&data);
        switch (dimensions) {
            case 1:
                initialize_data_1d(&arr_data);
                jacobi1d_sequential(&arr_data);
                break;
            case 2:
                initialize_data_2d(&arr_data);
                jacobi2d_sequential(&arr_data);
                break;
            case 3:
                initialize_data_3d(&arr_data);
                jacobi3d_sequential(&arr_data);
                break;
        }
    } else {
        // Add CUDA calls
    }
}