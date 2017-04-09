#include "matrix.h"

__global__ void jacobi1d_naive(Matrix data, Matrix result) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float newValue;

    if (id > 0 && id < size - 1) {
        newValue = (data.elements[id - 1] + data.elements[id] + data.elements[id + 1]) / 3;
    } else if (id == 0 || id == size - 1) {
        // Edge, do not change.
        newValue = data.elements[id];
    } else {
        /* TODO: Test if this is a necessary condition */
        // Beyond edge, just in case.
        newValue = 0.0;
    }

    __syncthreads();

    if (id < size) {
        result.elements[id] = tmp;
    }
}

__global__ void jacobi2d_naive(Matrix data, Matrix result) {
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int x = blockCol * TILE_WIDTH + threadCol;
    int y = blockRow * TILE_HEIGHT + threadRow;

    int index = x + y * data.width;
    int xPrev = (x - 1) + y * data.width;
    int xNext = (x + 1) + y * data.width;
    int yPrev = x + (y - 1) * data.width;
    int yNext = x + (y + 1) * data.width;

    float tmp;

    if (x > 0 && x < data.width - 1 && y > 0 && y < data.height - 1) {
        tmp =
            (
                data.elements[index] +
                data.elements[xPrev] +
                data.elements[xNext] +
                data.elements[yPrev] +
                data.elements[yNext]
            ) * 0.2;
    } else if (x == 0 || x == data.width - 1 || y == 0 || y == data.height - 1) {
        // Edge, do not change.
        tmp = data[id];
    } else {
        /* TODO: Test if this is a necessary condition */
        // Beyond the edge. We should be avoiding it, but just in case.
        tmp = 0.0;
    }

    __syncthreads();

    if (x < data.width && y < data.height) {
        result.elements[index] = tmp;
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

    float tmp;

    if (x > 0 && x < data.width - 1 && y > 0 && y < data.height - 1 && z > 0 && z < data.depth - 1) {
        tmp =
            (
                data.elements[index] +
                data.elements[xPrev] +
                data.elements[xNext] +
                data.elements[yPrev] +
                data.elements[yNext] +
                data.elements[zPrev] +
                data.elements[zNext]
            ) / 7;
    } else if (x == 0 || x == data.width - 1 || y == 0 || y == data.height - 1 || z == 0 || z == data.depth - 1) {
        // Edge, do not change.
        tmp = data[id];
    } else {
        /* TODO: Test if this is a necessary condition */
        // Beyond the edge. We should be avoiding it, but just in case.
        tmp = 0.0;
    }

    __syncthreads();

    if (x < data.width && y < data.height && z < data.depth) {
        result.elements[index] = tmp;
    }
}