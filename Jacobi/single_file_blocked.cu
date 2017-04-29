#include <stdio.h>
#include <iostream>
#include <unistd.h>

using namespace std;

// Shorthand for formatting and printing usage options to stderr
#define fpe(msg) fprintf(stderr, "\t%s\n", msg);

// Shorthand for handling CUDA errors.
#define HANDLE_ERROR(err)  ( HandleError( err, __FILE__, __LINE__ ) )

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
    bool debug = false;
    bool sequential = false;
    bool blocked = false;
    bool overlapped = false;
    // Data attributes
    int size = 1024, dimensions = 2, alloc_size;
    int xSize = 1, ySize = 1, zSize = 1;
    int xBlockSize = 1, yBlockSize = 1, zBlockSize = 1, tBlockSize;
    // Run attributes
    int grid_size = 1, block_count = -1, thread_count = -1, iterations = 1000;
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
    float *elements;
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

#define TILE_WIDTH 4
#define TILE_HEIGHT 2
#define TILE_DEPTH 2
#define PER_THREAD_X 2
#define PER_THREAD_Y 2
#define PER_THREAD_Z 2

__global__ void jacobi1d(Matrix data, Matrix result) {
    int threadCol = threadIdx.x;
    int blockCol = blockIdx.x;

    // Where the block starts in global data
    int blockStart = blockCol * TILE_WIDTH;
    //// Where the thread starts in the shared block
    //int threadStart = threadCol * PER_THREAD;

    __shared__ float shared[TILE_WIDTH];
    float local[PER_THREAD_X];

#pragma unroll
    for (int i = 0; i < PER_THREAD_X; i++) {
        // Issue contiguous reads, e.g. for 4 threads, 2 per thread: do 11|11|22|22 instead of 12|12|12|12
        // => shared[ [0-3] + 4 * [0-1] ]= elements[ [0-3] + 4 * [0-1] + blockStart ]
        shared[threadCol + blockDim.x * i] = data.elements[threadCol + blockDim.x * i + blockStart];
    }

#pragma unroll
    for (int i = 0; i < PER_THREAD_X; i++) {
        int x = threadCol + i * blockDim.x;
        int globalX = x + blockStart;
        if (globalX > 0 && globalX < data.width - 1) {
            local[i] = (shared[x] + shared[x - 1] + shared[x + 1]) / 3;
        } else if (globalX == 0 || globalX == data.width - 1) {
            local[i] = shared[x];
        } else {
            // Outside of the bounds
        }
    }

    __syncthreads();

#pragma unroll
    for (int i = 0; i < PER_THREAD_X; i++) {
        int x = threadCol + blockDim.x * i + blockStart;
        if (x < data.width) {
            result.elements[x] = local[i];
        }
    }
}

__global__ void jacobi2d(Matrix data, Matrix result) {
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Indexes so we don't have to recompute them.
    int globalIndex[PER_THREAD_Y][PER_THREAD_X];
    int globalX[PER_THREAD_X];
    int globalY[PER_THREAD_Y];
    int sharedX[PER_THREAD_X];
    int sharedY[PER_THREAD_Y];

    // Shared and local data arrays
    __shared__ float shared[TILE_HEIGHT + 2][TILE_WIDTH + 2];
    float local[PER_THREAD_Y][PER_THREAD_X];

    /*
     * Calculate indexes into the global and shared arrays
     */

    // X shared and global
#pragma unroll
    for (int x = 0; x < PER_THREAD_X; x++) {
        sharedX[x] = threadCol + blockDim.x * x + 1;
        globalX[x] = blockCol * TILE_WIDTH + sharedX[x] - 1;
    }

    // Y shared and global
#pragma unroll
    for (int y = 0; y < PER_THREAD_Y; y++) {
        sharedY[y] = threadRow + blockDim.y * y + 1;
        globalY[y] = blockRow * TILE_HEIGHT + sharedY[y] - 1;
    }

    // Global absolute index
#pragma unroll
    for (int y = 0; y < PER_THREAD_Y; y++) {
#pragma unroll
        for (int x = 0; x < PER_THREAD_X; x++) {
            globalIndex[y][x] = globalX[x] + globalY[y] * data.width;
        }
    }

    /*
     * Copy into shared memory
     */
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
            shared[sharedY[y]][sharedX[x]] = data.elements[globalIndex[y][x]];
        }
    }

    // Copy below-block dependencies into shared memory
    if (threadRow == 0 && blockRow > 0) {
#pragma unroll
        for (int x = 0; x < PER_THREAD_X; x++) {
            shared[0][sharedX[x]] = data.elements[globalIndex[0][x] - data.width];
        }
    }

    // Copy above-block dependencies into shared memory
    if (threadRow == blockDim.y - 1 && (blockRow + 1) * TILE_HEIGHT < data.height - 1) {
#pragma unroll
        for (int x = 0; x < PER_THREAD_X; x++) {
            shared[TILE_HEIGHT + 1][sharedX[x]] = data.elements[globalIndex[PER_THREAD_Y - 1][x] + data.width];
        }
    }

    // Copy left-of-block dependencies into shared memory
    if (threadCol == 0 && blockCol > 0) {
#pragma unroll
        for (int y = 0; y < PER_THREAD_Y; y++) {
            shared[sharedY[y]][0] = data.elements[globalIndex[y][0] - 1];
        }
    }

    // Copy right-of-block dependencies into shared memory
    if (threadCol == blockDim.x - 1 && (blockCol + 1) * TILE_WIDTH < data.width) {
#pragma unroll
        for (int y = 0; y < PER_THREAD_Y; y++) {
            shared[sharedY[y]][TILE_WIDTH + 1] = data.elements[globalIndex[y][PER_THREAD_X - 1] + 1];
        }
    }

    __syncthreads();

    /*
     * Calculate Values
     */
#pragma unroll
    for (int y = 0; y < PER_THREAD_Y; y++) {
        int globY = globalY[y];
        int sharY = sharedY[y];
#pragma unroll
        for (int x = 0; x < PER_THREAD_X; x++) {
            int globX = globalX[x];
            int sharX = sharedX[x];

            if (globX > 0 && globX < data.width - 1 && globY > 0 && globY < data.height - 1) {
                // Calculate new value
                local[y][x] =
                    (
                        shared[sharY][sharX - 1] +
                        shared[sharY][sharX] +
                        shared[sharY][sharX + 1] +
                        shared[sharY - 1][sharX] +
                        shared[sharY + 1][sharX]
                    ) * 0.2;
            } else if (globX == 0 || globX == data.width - 1 || globY == 0 || globY == data.height - 1) {
                // On the edge
                local[y][x] = shared[sharY][sharX];
            } else {
                // Beyond the edge, shouldn't ever hit this unless we messed something up
            }
        }
    }

    __syncthreads();

#pragma unroll
    for (int y = 0; y < PER_THREAD_Y; y++) {
#pragma unroll
        for (int x = 0; x < PER_THREAD_X; x++) {
            result.elements[globalIndex[y][x]] = local[y][x];
        }
    }
}

__global__ void jacobi3d(Matrix data, Matrix result) {
    // TODO
}

/********************
 * END CUDA KERNELS *
 ********************/

Matrix initialize_device(Matrix A, bool copyToDevice) {
    Matrix deviceA;

    deviceA.width = A.width;
    deviceA.height = A.height;
    deviceA.depth = A.depth;
    deviceA.dimensions = A.dimensions;

    size_t sizeA = A.width * A.height * A.depth * sizeof(float);

    HANDLE_ERROR(cudaMalloc((void **) &deviceA.elements, sizeA));
    if (copyToDevice) {
        HANDLE_ERROR(cudaMemcpy(deviceA.elements, A.elements, sizeA, cudaMemcpyHostToDevice));
    }

    return deviceA;
}

void callKernel(Args args, Matrix A, Matrix B) {
    Matrix deviceA, deviceB;

    deviceA = initialize_device(A, true);
    deviceB = initialize_device(B, false);

    if (args.dimensions == 1) {
        dim3 blocks(max(args.size / (args.xBlockSize / PER_THREAD_X) , 1));
        dim3 threads(args.xBlockSize);

        for (int t = 0; t < args.iterations; t++) {
            jacobi1d<<<blocks, threads>>>(deviceA, deviceB);
            swap(deviceA, deviceB);
        }
    } else if (args.dimensions == 2) {
        dim3 blocks(max(args.size / TILE_WIDTH, 1), max(args.size / TILE_HEIGHT, 1));
        dim3 threads(TILE_WIDTH / PER_THREAD_X, TILE_HEIGHT / PER_THREAD_Y);
        for (int t = 0; t < args.iterations; t++) {
            jacobi2d<<<blocks, threads>>>(deviceA, deviceB);
            checkCUDAError("jacobi2d", true);
            swap(deviceA, deviceB);
        }
    } else {
        dim3 blocks(max(args.size / TILE_WIDTH, 1), max(args.size / TILE_HEIGHT, 1), max(args.size / TILE_DEPTH, 1));
        dim3 threads(TILE_WIDTH / PER_THREAD_X, TILE_HEIGHT / PER_THREAD_Y, TILE_DEPTH / PER_THREAD_Z);
        for (int t = 0; t < args.iterations; t++) {
            jacobi3d<<<blocks, threads>>>(deviceA, deviceB);
            swap(deviceA, deviceB);
        }
    }

    HANDLE_ERROR(cudaMemcpy(B.elements, deviceA.elements, A.width * A.height * A.depth * sizeof(float), cudaMemcpyDeviceToHost));
}

// Data output
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

// Main
int main(int argc, char *argv[]) {
    Args args = parse_arguments(argc, argv);
    Matrix A, B;
    A = initialize_matrix(args.dimensions, args.size, args.size, args.size);
    B = initialize_matrix(args.dimensions, args.size, args.size, args.size);

    atexit(cleanupCuda);

    //if (args.debug) { print_data(data, args.size, args.dimensions); }
    callKernel(args, A, B);
    if (args.debug) { print_data(B.elements, args.size, args.dimensions); }
}
