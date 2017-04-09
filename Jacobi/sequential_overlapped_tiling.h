//
// Created by alex on 4/2/17.
//

#ifndef CSC5593_GPU_PROJECT_SEQUENTIAL_OVERLAPPED_TILING_H
#define CSC5593_GPU_PROJECT_SEQUENTIAL_OVERLAPPED_TILING_H

#include <algorithm>
#include <cmath>

#include <iostream>

using namespace std;

void
jacobi1d_sequential_overlapped(float *data, float *temp, int iterations, int size, int xBlockSize, int tBlockSize) {
    int xxIterations = size / xBlockSize;
    int ttIterations = (int) ceil((float) iterations / (float) tBlockSize);
    int xBlockStart, xBlockEnd;
    int xStart, xEnd;

    for (int tt = 0; tt < ttIterations; tt++) {
        for (int xx = 0; xx < xxIterations; xx++) {
            // 0 or block beginning - t-height of block
            xBlockStart = xx * xBlockSize - tBlockSize;
            xBlockEnd = (xx + 1) * xBlockSize + tBlockSize;

            for (int t = 0; t < tBlockSize; t++) {
                xStart = max(xBlockStart + t, 1);
                xEnd = min(xBlockEnd - t, size - 1);
                for (int x = xStart; x < xEnd; x++) {
                    temp[x] = (data[x] + data[x - 1] + data[x + 1]) / 3;
                }
            }

            swap(data, temp);
        }
    }
}

void jacobi2d_sequential_overlapped(float *data, float *temp, int iterations, int size, int xBlockSize, int yBlockSize,
                                    int tBlockSize) {
    int xxIterations = (int) ceil((float) size / (float) xBlockSize);
    int yyIterations = (int) ceil((float) size / (float) yBlockSize);
    int ttIterations = (int) ceil((float) size / (float) tBlockSize);

    int yBlockStart, yBlockEnd, xBlockStart, xBlockEnd, yStart, yEnd, xStart, xEnd;

    float *read;
    float *write;

    for (int tt = 0; tt < ttIterations; tt++) {
        for (int yy = 0; yy < yyIterations; yy++) {
            yBlockStart = yy * yBlockSize - tBlockSize + 1;
            yBlockEnd = (yy + 1) * yBlockSize + tBlockSize - 1;
            for (int xx = 0; xx < xxIterations; xx++) {
                xBlockStart = xx * xBlockSize - tBlockSize + 1;
                xBlockEnd = (xx + 1) * xBlockSize + tBlockSize - 1;

                for (int t = 0; t < tBlockSize; t++) {
                    read = temp + size * size * (t-1);
                    write = temp + size * size * t;
                    if (t == 0) {
                        read = data;
                    } else if (t == tBlockSize - 1) {
                        write = data;
                    }

                    yStart = max(yBlockStart + t, 1);
                    yEnd = min(yBlockEnd - t, size - 1);
                    for (int y = yStart; y < yEnd; y++) {
                        xStart = max(xBlockStart + t, 1);
                        xEnd = min(xBlockEnd - t, size - 1);

                        for (int x = xStart; x < xEnd; x++) {
                            fprintf(stdout, "%d %d %d %d %d %d\n", tt, yy, xx, t, y, x);
                            write[y * size + x] =
                                (
                                    read[y * size + x] +
                                    read[(y - 1) * size + x] +
                                    read[(y + 1) * size + x] +
                                    read[y * size + x - 1] +
                                    read[y * size + x + 1]
                                ) / 5;
                        }
                    }
                }
            }
        }
    }
}

void jacobi3d_sequential_overlapped(float *data, float *temp, int iterations, int size, int xBlockSize, int yBlockSize,
                                    int zBlockSize, int tBlockSize) {
    int xxIterations = (int) ceil((float) size / (float) xBlockSize);
    int yyIterations = (int) ceil((float) size / (float) yBlockSize);
    int zzIterations = (int) ceil((float) size / (float) zBlockSize);
    int ttIterations = (int) ceil((float) iterations / (float) tBlockSize);

    int xBlockStart, xBlockEnd, yBlockStart, yBlockEnd, zBlockStart, zBlockEnd;
    int xStart, xEnd, yStart, yEnd, zStart, zEnd;

    for (int tt = 0; tt < ttIterations; tt++) {
        for (int zz = 0; zz < zzIterations; zz++) {
            int zBlockStart = zz * zBlockSize, zBlockEnd = (zz + 1) * zBlockSize;
            for (int yy = 0; yy < yyIterations; yy++) {
                int yBlockStart = yy * yBlockSize, yBlockEnd = (yy + 1) * yBlockSize;
                for (int xx = 0; xx < xxIterations; xx++) {
                    int xBlockStart = xx * xBlockSize, xBlockEnd = (xx + 1) * xBlockSize;
                    for (int t = 0; t < tBlockSize; t++) {
                        zStart = max(zBlockStart + t, 1);
                        zEnd = min(zBlockEnd - t, size - 1);
                        for (int z = zStart; z < zEnd; z++) {
                            yStart = max(yBlockStart + t, 1);
                            yEnd = min(yBlockEnd - t, size - 1);
                            for (int y = yStart; y < yEnd; y++) {
                                xStart = max(xBlockStart + t, 1);
                                xEnd = min(xBlockEnd - t, size - 1);
                                for (int x = xStart; x < xEnd; x++) {
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

                        swap(data, temp);
                    }
                }
            }
        }

    }
}

void jacobi_sequential_overlapped(float *data, float *temp, int iterations, int size, int dimensions, int tBlockSize,
                                  int xBlockSize, int yBlockSize = 0, int zBlockSize = 0) {
    switch (dimensions) {
        case 1:
            jacobi1d_sequential_overlapped(data, temp, iterations, size, xBlockSize, tBlockSize);
            break;
        case 2:
            jacobi2d_sequential_overlapped(data, temp, iterations, size, xBlockSize, yBlockSize, tBlockSize);
            break;
        case 3:
            jacobi3d_sequential_overlapped(data, temp, iterations, size, xBlockSize, yBlockSize, zBlockSize,
                                           tBlockSize);
            break;
    }
}

#endif //CSC5593_GPU_PROJECT_SEQUENTIAL_OVERLAPPED_TILING_H
