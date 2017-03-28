//
// Created by alex on 3/28/17.
//

#ifndef CSC5593_GPU_PROJECT_SEQUENTIAL_H
#define CSC5593_GPU_PROJECT_SEQUENTIAL_H

#include <algorithm>

using namespace std;

void jacobi1d_sequential(float *data, float *temp, int iterations, int size) {
    for (int t = 0; t < iterations; t++) {
        for (int x = 1; x < size - 1; x++) {
            // Include Self
            temp[x] = (data[x] + data[x - 1] + data[x + 1]) / 3;
            // Exclude Self
            //temp[x] = (data[x - 1] + data[x + 1]) / 2;
        }

        swap(data, temp);
    }
}

void jacobi2d_sequential(float *data, float *temp, int iterations, int size) {
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
        swap(data, temp);
    }
}

void jacobi3d_sequential(float *data, float *temp, int iterations, int size) {
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

        swap(data, temp);
    }
}

void jacobi_sequential(float *data, float *temp, int iterations, int size, int dimensions) {
    switch (dimensions) {
        case 1:
            jacobi1d_sequential(data, temp, iterations, size);
            break;
        case 2:
            jacobi2d_sequential(data, temp, iterations, size);
            break;
        case 3:
            jacobi3d_sequential(data, temp, iterations, size);
            break;
    }
}

#endif //CSC5593_GPU_PROJECT_SEQUENTIAL_H
