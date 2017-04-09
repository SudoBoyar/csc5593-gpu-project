__global__ void jacobi1d_naive(float *data) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp;
    if (id > 0 && id < size) {
        tmp = (data[id - 1] + data[id] + data[id + 1]) / 3;
    } else {
        // Edge, do not change.
        tmp = data[id];
    }
    // Note: this sync is to prevent RAW issues inside of blocks. There is currently nothing preventing it between
    // blocks.
    __syncthreads();

    data[id] = tmp;
}

__global__ void jacobi2d_naive() {
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