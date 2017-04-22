#include "matrix.h"

__global__ void jacobi1d_naive(Matrix data, Matrix result) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float newValue;

    if (id > 0 && id < size - 1) {
        newValue = (data.elements[id - 1] + data.elements[id] + data.elements[id + 1]) / 3;
        __syncthreads();
        result.elements[id] = newValue;
    } else {
        // Edge or outside completely, do not change.
        __syncthreads();
    }
}

__global__ void jacobi2d_naive(Matrix data, Matrix result) {
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int x = blockCol * blockDim.x + threadCol;
    int y = blockRow * blockDim.y + threadRow;

    int index = x + y * data.width;
    int xPrev = (x - 1) + y * data.width;
    int xNext = (x + 1) + y * data.width;
    int yPrev = x + (y - 1) * data.width;
    int yNext = x + (y + 1) * data.width;

    float newValue;

    if (x > 0 && x < data.width - 1 && y > 0 && y < data.height - 1) {
        newValue =
            (
                data.elements[index] +
                data.elements[xPrev] +
                data.elements[xNext] +
                data.elements[yPrev] +
                data.elements[yNext]
            ) * 0.2;
        __syncthreads();
        result.elements[index] = newValue;
    } else {
        // Edge or beyond, do not change.
        __syncthreads();
    }
}

__global__ void jacobi3d_naive(Matrix data, Matrix result) {
    int threadDep = threadIdx.z;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    int blockDep = blockIdx.z;
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int x = blockCol * TILE_WIDTH + threadCol;
    int y = blockRow * TILE_HEIGHT + threadRow;
    int z = blockDep * TILE_DEPTH + threadDep;

    int xySurface = data.width * data.height;
    int zTemp = z * xySurface;
    int yTemp = y * data.width;

    int index = x + yTemp + zTemp; // x + y * data.width + z * data.width * data.height;
    int xPrev = (x - 1) + yTemp + zTemp; // (x-1) + y * data.width + z * data.width * data.height;
    int xNext = (x + 1) + yTemp + zTemp; // (x+1) + y * data.width + z * data.width * data.height;
    int yPrev = x + yTemp - data.width + zTemp; // x + (y-1) * data.width + z * data.width * data.height;
    int yNext = x + yTemp + data.width + zTemp; // x + (y+1) * data.width + z * data.width * data.height;
    int zPrev = x + yTemp + zTemp - xySurface; // x + y * data.width + (z-1) * data.width * data.height;
    int zNext = x + yTemp + zTemp + xySurface; // x + y * data.width + (z+1) * data.width * data.height;

    float newValue;

    if (x > 0 && x < data.width - 1 && y > 0 && y < data.height - 1 && z > 0 && z < data.depth - 1) {
        newValue =
            (
                data.elements[index] +
                data.elements[xPrev] +
                data.elements[xNext] +
                data.elements[yPrev] +
                data.elements[yNext] +
                data.elements[zPrev] +
                data.elements[zNext]
            ) / 7;
        __syncthreads();
        result.elements[index] = newValue;
    } else {
        // Edge or beyond, do not change.
        __syncthreads();
    }
}