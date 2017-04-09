//
// Created by alex on 4/4/17.
//

#ifndef CSC5593_GPU_PROJECT_MATRIX_H
#define CSC5593_GPU_PROJECT_MATRIX_H

typedef struct {
    int dimensions;
    int height;
    int width;
    int depth;
    float* elements;
} Matrix;

#endif //CSC5593_GPU_PROJECT_MATRIX_H
