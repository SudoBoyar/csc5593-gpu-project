//
// Created by alex on 4/2/17.
//

#ifndef CSC5593_GPU_PROJECT_SEQUENTIAL_BLOCKED_H
#define CSC5593_GPU_PROJECT_SEQUENTIAL_BLOCKED_H

#include <algorithm>

using namespace std;

void jacobi1d_sequential_blocked(float *data, float *temp, int iterations, int size, int xBlockSize) {
    int xxIterations = size / xBlockSize;

    for (int t = 0; t < iterations; t++) {
        for (int xx = 0; xx < xxIterations; xx++) {
            int xBlockStart = xx * xBlockSize, xBlockEnd = (xx + 1) * xBlockSize;
            for (int x = xBlockStart; x < xBlockEnd; x++) {
                if (x == 0 || x == size - 1) {
                    continue;
                }

                temp[x] = (data[x] + data[x - 1] + data[x + 1]) / 3;
            }
        }

        swap(data, temp);
    }
}

void jacobi2d_sequential_blocked(float *data, float *temp, int iterations, int size, int xBlockSize, int yBlockSize) {
    int xxIterations = size / xBlockSize, yyIterations = size / yBlockSize;

    for (int t = 0; t < iterations; t++) {
        for (int yy = 0; yy < yyIterations; yy++) {
            int yBlockStart = yy * yBlockSize, yBlockEnd = (yy + 1) * yBlockSize;
            for (int xx = 0; xx < xxIterations; xx++) {
                int xBlockStart = xx * xBlockSize, xBlockEnd = (xx + 1) * xBlockSize;
                for (int y = yBlockStart; y < yBlockEnd; y++) {
                    if (y == 0 || y == size - 1) {
                        continue;
                    }
                    for (int x = xBlockStart; x < xBlockEnd; x++) {
                        if (x == 0 || x == size - 1) {
                            continue;
                        }
                        temp[y * size + x] =
                            (
                                data[y * size + x] +
                                data[(y - 1) * size + x] +
                                data[(y + 1) * size + x] +
                                data[y * size + x - 1] +
                                data[y * size + x + 1]
                            ) / 5;
                    }
                }
            }
        }
        swap(data, temp);
    }
}

void jacobi3d_sequential_blocked(float *data, float *temp, int iterations, int size, int xBlockSize, int yBlockSize,
                                 int zBlockSize) {
    int xxIterations = size / xBlockSize, yyIterations = size / yBlockSize, zzIterations = size / zBlockSize;
    for (int t = 0; t < iterations; t++) {
        for (int zz = 0; zz < zzIterations; zz++) {
            int zBlockStart = zz * zBlockSize, zBlockEnd = (zz + 1) * zBlockSize;
            for (int yy = 0; yy < yyIterations; yy++) {
                int yBlockStart = yy * yBlockSize, yBlockEnd = (yy + 1) * yBlockSize;
                for (int xx = 0; xx < xxIterations; xx++) {
                    int xBlockStart = xx * xBlockSize, xBlockEnd = (xx + 1) * xBlockSize;
                    for (int z = zBlockStart; z < zBlockEnd; z++) {
                        if (z == 0 || z == size - 1) {
                            continue;
                        }
                        for (int y = yBlockStart; y < yBlockEnd; y++) {
                            if (y == 0 || y == size - 1) {
                                continue;
                            }
                            for (int x = xBlockStart; x < xBlockEnd; x++) {
                                if (x == 0 || x == size - 1) {
                                    continue;
                                }
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
                            }
                        }
                    }

                }
            }
        }

        swap(data, temp);
    }
}

void jacobi_sequential_blocked(float *data, float *temp, int iterations, int size, int dimensions, int xBlockSize,
                               int yBlockSize = 0, int zBlockSize = 0) {
    switch (dimensions) {
        case 1:
            jacobi1d_sequential_blocked(data, temp, iterations, size, xBlockSize);
            break;
        case 2:
            jacobi2d_sequential_blocked(data, temp, iterations, size, xBlockSize, yBlockSize);
            break;
        case 3:
            jacobi3d_sequential_blocked(data, temp, iterations, size, xBlockSize, yBlockSize, zBlockSize);
            break;
    }
}

#endif //CSC5593_GPU_PROJECT_SEQUENTIAL_BLOCKED_H
