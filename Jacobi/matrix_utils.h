//
// Created by alex on 4/4/17.
//

#ifndef CSC5593_GPU_PROJECT_MATRIX_UTILS_H
#define CSC5593_GPU_PROJECT_MATRIX_UTILS_H

#include "matrix.h"

Matrix initialize_matrix(int dimensions, int width, int height = 1, int depth = 1) {
    Matrix data;

    if (dimension == 1 && width > 1) {
        data.width = width;
        data.height = 1;
        data.depth = 1;
        data.elements = (float *) malloc(width * sizeof(float));

        data.elements[0] = 1.0;
        data.elements[width - 1] = 1.0;
    } else if (dimensions == 2 && width > 1 && height > 1) {
        data.width = width;
        data.height = height;
        data.depth = 1;
        data.elements = (float *) malloc(width * height * sizeof(float));

        for (int y = 0; y < height; y += height - 1) {
            for (int x = 0; x < width; x++) {
                data.elements[y * width + x] = 1.0;
            }
        }

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x += width - 1) {
                data.elements[y * width + x] = 1.0;
            }
        }
    } else if (dimensions == 3 && width > 1 && height > 1 && depth > 1) {
        data.width = width;
        data.height = height;
        data.depth = depth;
        data.elements = (float *) malloc(width * height * depth * sizeof(float));

        for (int z = 0; z < depth; z++) {
            // X = 0 & N planes
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x += width - 1) {
                    data.elements[z * width * height + y * width + x] = 1.0;
                }
            }

            // Y = 0 & N planes
            for (int y = 0; y < height; y += height - 1) {
                for (int x = 0; x < width; x++) {
                    data.elements[z * width * height + y * width + x] = 1.0;
                }
            }
        }

        // Z = 0 & N planes
        for (int z = 0; z < depth; z += depth - 1) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    data.elements[z * width * height + y * width + x] = 1.0;
                }
            }
        }
    } else {
        // Bad data
    }

    return data;
}

Matrix copy_matrix(Matrix source) {
    return initialize_matrix(source.dimensions, source.width, source.height, source.depth);
    //Matrix target;
    //target.dimensions = source.dimensions;
    //target.width = source.width;
    //target.height = source.height;
    //target.depth = source.depth;
    //target.elements = (float *) malloc(target.width * target.height * target.depth * sizeof(float));
    //
    //for (int x = 0; x < source.width * source.height * source.depth; x++) {
    //    target.elements[x] = source.elements[x];
    //}
    //
    //return target;
}


#endif //CSC5593_GPU_PROJECT_MATRIX_UTILS_H
