#include "matrix.h"


__global__ void jacobi1d_blocked(Matrix data, Matrix result) {
    int threadCol = threadIdx.x;
    int blockCol = blockIdx.x;

    // Where the block starts in global data
    int blockStart = blockCol * TILE_WIDTH;
    //// Where the thread starts in the shared block
    //int threadStart = threadCol * PER_THREAD;

    __shared__ float shared[TILE_WIDTH];
    float local[PER_THREAD];

#pragma unroll
    for (int i = 0; i < PER_THREAD; i++) {
        // Issue contiguous reads, e.g. for 4 threads, 2 per thread: do 11|11|22|22 instead of 12|12|12|12
        // => shared[ [0-3] + 4 * [0-1] ]= elements[ [0-3] + 4 * [0-1] + blockStart ]
        shared[threadCol + threadDim.x * i] = data.elements[threadCol + threadDim.x * i + blockStart];
    }

#pragma unroll
    for (int i = 0; i < PER_THREAD; i++) {
        int x = threadCol + i * threadDim.x;
        int globalX = x + blockStart;
        if (globalX > 1 && globalX < data.width - 1) {
            local[i] = (shared[x] + shared[x - 1] + shared[x + 1]) / 3;
        } else if (globalX == 0 || globalX == data.width - 1) {
            local[i] = shared[x];
        } else {
            local[i] = 0.0;
        }
    }

    __syncthreads();

#pragma unroll
    for (int i = 0; i < PER_THREAD; i++) {
        int x = threadCol + threadDim.x * i + blockStart;
        if (x < data.width) {
            result.elements[x] = local[i];
        }
    }
}

__global__ void jacobi2d_blocked(Matrix data, Matrix result) {
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Tile Starting
    int xTileStart = blockCol * TILE_WIDTH;
    int yTileStart = blockRow * TILE_HEIGHT;

    __shared__ float shared[TILE_WIDTH][TILE_HEIGHT];
    float local[PER_THREAD_X][PER_THREAD_Y];

#pragma unroll
    for (int y = 0; y < PER_THREAD_Y; y++) {
#pragma unroll
        for (int x = 0; x < PER_THREAD_X; x++) {
            /*
             * We want to be doing block-contiguous reads, e.g. for 2x2 block dimension, 2 per thread for x and y
             * we want the read pattern to look like:
             *
             * 11|22
             * 11|22
             * -----
             * 33|44
             * 33|44
             *
             * Optimizing the width for reads is the responsibility of the calling code.
             */
            // TODO: Index integrity checks for out of tile range.
            shared[threadCol + threadDim.x * x][threadRow + threadDim.y * y] =
                data.elements[
                    yTileStart + // Y location of tile start in data
                    threadRow * data.width + // Up to the thread's initial row
                    threadDim.y * y * data.width + // And up again to get to the y'th sub-block
                    xTileStart + // X location of tile start in data
                    threadCol + // Over to the initial x position for the thread
                    threadDim.x * x // And over again to skip to the x'th sub_block
                ];
        }
    }

#pragma unroll
    for (int y = 0; y < PER_THREAD_Y; y++) {
        int globalY = yTileStart + threadRow * data.width + threadDim.y * y * data.width;
        int sharedY = threadRow + threadDim.y * y;
#pragma unroll
        for (int x = 0; x < PER_THREAD_X; x++) {
            int globalX = xTileStart + threadCol + threadDim.x * x;
            int sharedX = threadCol + x * threadDim.x;
            if (globalX > 0 && globalX < data.width - 1 && globalY > 0 && globalY < data.height - 1) {
                // Calculate new value
                local[x][y] =
                    (
                        shared[sharedX][sharedY] +
                        shared[sharedX - 1][sharedY] +
                        shared[sharedX + 1][sharedY] +
                        shared[sharedX][sharedY - 1] +
                        shared[sharedX][sharedY + 1]
                    ) * 0.2;
            } else if (globalX == 0 || globalX == data.width - 1 || globalY == 0 || globalY == data.height - 1) {
                // Edge, do not change.
                local[x][y] = shared[sharedX][sharedY];
            } else {
                /* TODO: Test if this is a necessary condition */
                // Beyond the edge. We should be avoiding it, but just in case.
                local[x][y] = 0.0;
            }
        }
    }

    __syncthreads();


#pragma unroll
    for (int y = 0; y < PER_THREAD_Y; y++) {
#pragma unroll
        for (int x = 0; x < PER_THREAD_X; x++) {
            if (x < data.width && y < data.height) {
                result.elements[
                    yTileStart + // Y location of tile start in data
                    threadRow * data.width + // Up to the thread's initial row
                    threadDim.y * y * data.width + // And up again to get to the y'th sub-block
                    xTileStart + // X location of tile start in data
                    threadCol + // Over to the initial x position for the thread
                    threadDim.x * x // And over again to skip to the x'th sub_block
                ] = local[x][y];
            }
        }
    }
}

__global__ void jacobi3d_naive(float *data) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int x = id % size;
    int rem = id / size;
    int y = rem % size;
    int z = rem / size;
    float tmp;

    for (int t = 0; t < iterations; t++) {
        if (x > 0 && x < size - 1 && y > 0 && y < size - 1 && z > 0 && z < size - 1) {
            tmp =
                (
                    data[z * size * size + y * size + x] +
                    data[z * size * size + y * size + x - 1] +
                    data[z * size * size + y * size + x + 1] +
                    data[z * size * size + (y - 1) * size + x] +
                    data[z * size * size + (y + 1) * size + x] +
                    data[(z - 1) * size * size + y * size + x] +
                    data[(z + 1) * size * size + y * size + x]
                ) / 7;
        } else {
            // Edge, do not change.
            tmp = data[id];
        }

        // Note: this sync is to prevent RAW issues inside of blocks. There is currently nothing preventing it between
        // blocks.
        __syncthreads();

        data[id] = tmp;
    }
}