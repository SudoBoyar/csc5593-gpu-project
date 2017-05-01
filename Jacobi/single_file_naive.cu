
#include <stdio.h>
#include <iostream>
#include <unistd.h>

// Shorthand for formatting and printing usage options to stderr
#define fpe(msg) fprintf(stderr, "\t%s\n", msg);

// Shorthand for handling CUDA errors.
#define HANDLE_ERROR(err)  ( HandleError( err, __FILE__, __LINE__ ) )

using namespace std;

/*****************
 * CUDA Utilites *
 *****************/

void HandleError(cudaError_t err, const char *file, int line) {
    //
    // Handle and report on CUDA errors.
    //
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);

        exit(EXIT_FAILURE);
    }
}

void checkCUDAError(const char *msg, bool exitOnError) {
    //
    // Check cuda error and print result if appropriate.
    //
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        if (exitOnError) {
            exit(-1);
        }
    }
}

void cleanupCuda(void) {
    //
    // Clean up CUDA resources.
    //

    //
    // Explicitly cleans up all runtime-related resources associated with the
    // calling host thread.
    //
    HANDLE_ERROR(
        cudaThreadExit()
    );
}

/*********************
 * End CUDA Utilites *
 *********************/

struct Args {
    bool debug;
    bool sequential;
    bool blocked;
    bool overlapped;
    // Data attributes
    int size, dimensions, alloc_size;
    int xSize, ySize, zSize;
    int xBlockSize, yBlockSize, zBlockSize, tBlockSize;
    // Run attributes
    int grid_size, block_count, thread_count, iterations;
};

void usage(char *prog_name, string msg) {
    if (msg.size() > 0) {
        fputs(msg.c_str(), stderr);
    }

    fprintf(stderr, "%s\n", prog_name);
    fprintf(stderr, "Options are:\n");
    fpe("-n<size> Set data size (default: 1024)");
    fpe("-d<dims> Set number of data dimensions (1, 2, or 3) (default: 2)");
    fpe("-g<size> Set grid size");
    fpe("-b<num>  Set block count");
    fpe("-t<num>  Set thread count");
    fpe("-i<iter> Number of iterations to perform (default: 1000)");
    fpe("-x<size> X Dimension");
    fpe("-y<size> Y Dimension");
    fpe("-z<size> Z Dimension");
    fpe("-T<size> T Dimension");
    fpe("-S       Execute sequential, CPU version");
    fpe("-B       Execute blocked sequential, CPU version");
    fpe("-O       Execute sequential overlapped tiling, CPU version");
    fpe("-D       Print debug info");
    fpe("-h       Print usage info (this message)");
    exit(EXIT_FAILURE);
}

Args parse_arguments(int argc, char *argv[]) {
    Args args = Args();
    args.debug = false;
    args.sequential = false;
    args.blocked = false;
    args.overlapped = false;
    args.size = 1024;
    args.dimensions = 2;
    args.xSize = args.ySize = args.zSize = 1;
    args.xBlockSize = args.yBlockSize = args.zBlockSize = 1;
    args.grid_size = 1;
    args.block_count = -1;
    args.thread_count = -1;
    args.iterations = 1000;

    int opt;
    // Parse args
    while ((opt = getopt(argc, argv, "n:d:g:b:t:i:x:y:z:T:hSBOD")) != -1) {
        switch (opt) {
            case 'D':
                args.debug = true;
                break;
            case 'S':
                args.sequential = true;
                break;
            case 'B':
                args.blocked = true;
                break;
            case 'O':
                args.overlapped = true;
                break;
            case 'n':
                args.size = atoi(optarg);
                break;
            case 'd':
                args.dimensions = atoi(optarg);
                break;
            case 'g':
                args.grid_size = atoi(optarg);
                break;
            case 'b':
                args.block_count = atoi(optarg);
                break;
            case 't':
                args.thread_count = atoi(optarg);
                break;
            case 'i':
                args.iterations = atoi(optarg);
                break;
            case 'x':
                args.xBlockSize = atoi(optarg);
                break;
            case 'X':
                args.xSize = atoi(optarg);
                break;
            case 'y':
                args.yBlockSize = atoi(optarg);
                break;
            case 'Y':
                args.ySize = atoi(optarg);
                break;
            case 'z':
                args.zBlockSize = atoi(optarg);
                break;
            case 'Z':
                args.zSize = atoi(optarg);
                break;
            case 'T':
                args.tBlockSize = atoi(optarg);
                break;
            case 'h':
                usage(argv[0], "");
                break;
            default:
                usage(argv[0], "Unrecognized option\n");
        }
    }

    // check sizes
    if (args.size <= 0) {
        cout << "Data size must be larger than 0" << endl;
        exit(EXIT_FAILURE);
    }

    if (args.dimensions <= 0 || args.dimensions >= 4) {
        cerr << "Data must be 1, 2, or 3 dimensions" << endl;
        exit(EXIT_FAILURE);
    }

    // Calculations
    if (args.dimensions == 1) {
        args.alloc_size = args.size;
    } else if (args.dimensions == 2) {
        args.alloc_size = args.size * args.size;
    } else {
        args.alloc_size = args.size * args.size * args.size;
    }

    if (args.thread_count > 0) {
        args.block_count = args.alloc_size / args.thread_count;
    } else if (args.block_count > 0) {
        args.thread_count = args.alloc_size / args.block_count;
    } else {
        args.thread_count = 16;
        args.block_count = args.alloc_size / args.thread_count;
    }

    return args;
}

typedef struct {
    int dimensions;
    int height;
    int width;
    int depth;
    float* elements;
} Matrix;

Matrix initialize_matrix(int dimensions, int width, int height = 1, int depth = 1) {
    Matrix data;

    if (dimensions == 1 && width > 1) {
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
        fprintf(stderr, "Improper dimension or size.");
        exit(1);
    }

    return data;
}

/****************
 * CUDA KERNELS *
 ****************/

__global__ void jacobi1d(Matrix data, Matrix result) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float newValue;

    if (id > 0 && id < data.width - 1) {
        newValue = (data.elements[id - 1] + data.elements[id] + data.elements[id + 1]) / 3;
        __syncthreads();
        result.elements[id] = newValue;
    } else {
        // Edge or outside completely, do not change.
        __syncthreads();
    }
}

__global__ void jacobi2d(Matrix data, Matrix result) {
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

__global__ void jacobi3d(Matrix data, Matrix result) {
    int threadDep = threadIdx.z;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    int blockDep = blockIdx.z;
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int x = blockCol * blockDim.x + threadCol;
    int y = blockRow * blockDim.y + threadRow;
    int z = blockDep * blockDim.z + threadDep;

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

/********************
 * END CUDA KERNELS *
 ********************/

Matrix initialize_device(Matrix A) {
    Matrix deviceA;

    deviceA.width = A.width;
    deviceA.height = A.height;
    deviceA.depth = A.depth;
    deviceA.dimensions = A.dimensions;

    size_t sizeA = A.width * A.height * A.depth * sizeof(float);

    HANDLE_ERROR(cudaMalloc((void **) &deviceA.elements, sizeA));
    HANDLE_ERROR(cudaMemcpy(deviceA.elements, A.elements, sizeA, cudaMemcpyHostToDevice));

    return deviceA;
}

void callKernel(Args args, Matrix A, Matrix B) {
    Matrix deviceA, deviceB;
    deviceA = initialize_device(A);
    deviceB = initialize_device(B);

    if (args.dimensions == 1) {
        dim3 blocks(max(args.size/32, 1));
        dim3 threads(min(args.size, 32));

        for (int t = 0; t < args.iterations; t++) {
            jacobi1d<<<blocks, threads>>>(deviceA, deviceB);
//            checkCUDAError("jacobi1d", true);
            swap(deviceA, deviceB);
        }
    } else if (args.dimensions == 2) {
        dim3 blocks(max(args.size/16, 1), max(args.size/16, 1));
        dim3 threads(min(args.size, 16), min(args.size, 16));
        for (int t = 0; t < args.iterations; t++) {
            jacobi2d<<<blocks, threads>>>(deviceA, deviceB);
//            checkCUDAError("jacobi2d", true);
            swap(deviceA, deviceB);
        }
    } else {
        dim3 blocks(max(args.size/8, 1), max(args.size/8, 1), max(args.size/8, 1));
        dim3 threads(min(args.size, 8), min(args.size, 8), min(args.size, 8));
        for (int t = 0; t < args.iterations; t++) {
            jacobi3d<<<blocks, threads>>>(deviceA, deviceB);
//            checkCUDAError("jacobi3d", true);
            swap(deviceA, deviceB);
        }
    }

    cudaMemcpy(B.elements, deviceA.elements, A.width * A.height * A.depth * sizeof(float), cudaMemcpyDeviceToHost);
}

void print_data(float *data, int size, int dimensions) {
    if (size > 13) {
        cerr << "Data too big to print\n" << endl;
        return;
    }

    if (dimensions == 1) {
        for (int x = 0; x < size; x++) {
            printf("%.3f ", data[x]);
        }
    } else if (dimensions == 2) {
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                printf("%.3f ", data[y * size + x]);
            }
            cout << endl;
        }
    } else if (dimensions == 3) {
        for (int z = 0; z < size; z++) {
            for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                    printf("%.3f ", data[z * size * size + y * size + x]);
                }
                cout << endl;
            }
            cout << endl;
        }
    }
    cout << endl << endl;
}

int main(int argc, char *argv[]) {
    Args args = parse_arguments(argc, argv);
    Matrix A, B;
    A = initialize_matrix(args.dimensions, args.size, args.size, args.size);
    B = initialize_matrix(args.dimensions, args.size, args.size, args.size);

    float runtime;
    struct timeval start, end;

    gettimeofday(&start, NULL);
    callKernel(args, A, B);
    gettimeofday(&end, NULL);
    runtime = ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_usec - start.tv_usec) / 1000.0);
    printf("Processing Time: %4.4f milliseconds\n", runtime);
    if (args.debug) { print_data(B.elements, args.size, args.dimensions); }
}
