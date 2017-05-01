//
// Created by alex on 3/27/17.
//

#ifndef CSC5593_GPU_PROJECT_DATA_UTILS_H
#define CSC5593_GPU_PROJECT_DATA_UTILS_H

#include <iostream>

using namespace std;

void initialize_data_1d(float *data, int size) {
    // Left
    data[0] = 1.0;
    // Right
    data[size - 1] = 1.0;
}

void initialize_data_2d(float *data, int size) {
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

void initialize_data_3d(float *data, int size) {
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

void initialize_data(float *data, int size, int dimensions) {
    switch (dimensions) {
        case 1:
            initialize_data_1d(data, size);
            break;
        case 2:
            initialize_data_2d(data, size);
            break;
        case 3:
            initialize_data_3d(data, size);
            break;
    }
}


void print_data(float *data, int size, int dimensions) {
    if (size > 32) {
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

#endif //CSC5593_GPU_PROJECT_DATA_UTILS_H
