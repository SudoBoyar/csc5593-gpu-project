__global__ void jacobi1d_blocked(float *data, int iterations, int size) {
    int iterCount = iterations / blockDim.y;
    int iterStart = iterCount * blockIdx.y;
    int iterEnd = iterStart + iterCount;

    int tileSize = size / blockDim.x;
    int tileStart = tileSize * blockIdx.x;
    int tileEnd = tileStart + tileSize;

    int tid = threadIdx.x;

    int threadSize = tileSize / threadDim.x;

    __shared__ float temp[tileSize];
    float local[threadSize];

    float *dataPointer = data;

    for (int i = 0; i < threadSize; i++) {
        temp[threadSize * tid + i] = data[tileStart + threadSize * tid + i];
    }

    for (int t = iterStart; t < iterEnd; t++) {
        for (int i = 0; i < threadSize; i++) {
            local[i] = (dataPointer[tid * threadSize + i] +
                        dataPointer[tid * threadSize + i - 1] +
                        dataPointer[tid * threadSize + i + 1]) / 3;
        }
        __syncthreads();
        for (int i = 0; i < threadSize; i++) {
            temp[threadSize * tid + i] = local[i];
        }
    }

    for (int i = 0; i < threadSize; i++) {
        data[tileStart + threadSize * tid + i] = temp[threadSize * tid + i];
    }
}

__global__ void Jacobi2DKernel(Matrix A, Matrix result) {
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    float new_value = 0.0;
    int x = block_col * TILE_WIDTH + thread_col;
    int y = block_row * TILE_HEIGHT + thread_row;

    int A_index  =  x    +  y    * A.width;
    int A_x_prev = (x-1) +  y    * A.width;
    int A_x_next = (x+1) +  y    * A.width;
    int A_y_prev =  x    + (y-1) * A.width;
    int A_y_next =  x    + (y+1) * A.width;

    if (x >= A.width || y >= A.height) {
        new_value = 0.0;
    } else if (x == 0 || x == A.width - 1 ||  y == 0 || y == A.height - 1) {
        new_value = A.elements[A_index];
    } else {
        //if (0 < x  && x < (A.width-1) && 0 < y && y < (A.height-1)) {
        // 0.2*(A[i][j] + A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]);
        new_value = 0.2 * (A.elements[A_index]  +
                           A.elements[A_x_prev] +
                           A.elements[A_x_next] +
                           A.elements[A_y_prev] +
                           A.elements[A_y_next]
        );
    }

    __syncthreads();

    if (x < A.width && y < A.height) {
        result.elements[A_index] = new_value;
    }
}

__global__ void jacobi2d_naive() {
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    float new_value = 0.0;
    int x = block_col * TILE_WIDTH + thread_col;
    int y = block_row * TILE_HEIGHT + thread_row;

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int x = id % size;
    int y = id / size;
    float tmp;
    for (int t = 0; t < iterations; t++) {
        if (x > 0 && x < size - 1 && y > 0 && y < size - 1) {
            tmp =
                (
                    data[y * size + x] +
                    data[y * size + x - 1] +
                    data[y * size + x + 1] +
                    data[(y - 1) * size + x] +
                    data[(y + 1) * size + x]
                ) / 5;
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