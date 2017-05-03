#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <sys/time.h>

using namespace std;

// Shorthand for formatting and printing usage options to stderr
#define fpe(msg) fprintf(stderr, "\t%s\n", msg);

// Shorthand for handling CUDA errors.
#define HANDLE_ERROR(err)  ( HandleError( err, __FILE__, __LINE__ ) )

/**
 * DEFINED VALUES HERE
 */


#define TILE_WIDTH 32
#define TILE_HEIGHT 4
#define TILE_DEPTH 1
#define PER_THREAD_X 1
#define PER_THREAD_Y 1
#define PER_THREAD_Z 1

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
	HANDLE_ERROR(cudaThreadExit());
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

Matrix initialize_matrix(int dimensions, int width, int height = 1, int depth =
		1) {
	Matrix data;

	if (dimensions == 3 && width > 1 && height > 1 && depth > 1) {
		data.width = width;
		data.height = height;
		data.depth = depth;
		data.elements = (float *) malloc(
				width * height * depth * sizeof(float));

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


__global__ void cached_plane_shared_mem(Matrix data, Matrix result) {
	int threadCol = threadIdx.x;
	int threadRow = threadIdx.y;
	int threadDep = threadIdx.z;
	int blockCol = blockIdx.x;
	int blockRow = blockIdx.y;
	int blockDep = blockIdx.z;

	// Indexes so we don't have to recompute them.
	int globalIndex[PER_THREAD_Z][PER_THREAD_Y][PER_THREAD_X];
	int globalX[PER_THREAD_X];
	int globalY[PER_THREAD_Y];
	int globalZ[PER_THREAD_Z];
	int sharedX[PER_THREAD_X];
	int sharedY[PER_THREAD_Y];
	int sharedZ[PER_THREAD_Z];

	// Shared and local data arrays
	__shared__ float shared[TILE_DEPTH + 2][TILE_HEIGHT + 2][TILE_WIDTH + 2];
	//printf("here 1\n");
	float local[PER_THREAD_Z][PER_THREAD_Y][PER_THREAD_X];

//	printf("here 2\n");
	/*
	 * Calculate indexes into the global and shared arrays
	 */

	// X shared and global
#pragma unroll
	for (int x = 0; x < PER_THREAD_X; x++) {
		sharedX[x] = threadCol + blockDim.x * x + 1;
		globalX[x] = blockCol * TILE_WIDTH + sharedX[x] - 1;
	}

//	printf("here 3\n");
	// Y shared and global
#pragma unroll
	for (int y = 0; y < PER_THREAD_Y; y++) {
		sharedY[y] = threadRow + blockDim.y * y + 1;
		globalY[y] = blockRow * TILE_HEIGHT + sharedY[y] - 1;
	}

//	printf("here 4\n");
	// Z shared and global
#pragma unroll
	for (int z = 0; z < PER_THREAD_Z; z++) {
		sharedZ[z] = threadDep + blockDim.z * z + 1;
//		printf("tidx=%d,tidy=%d,tidz=%d, sharedZ[%d] = %d\n",threadIdx.x,threadIdx.y,threadIdx.z,z,sharedZ[z]);
		globalZ[z] = blockDep * TILE_DEPTH + sharedZ[z] - 1;
	}
//
//	printf("here 5\n");
//	printf("sizeof(sharedZ) = %d ", sizeof(sharedZ));
	// Global absolute index
#pragma unroll
	for (int z = 0; z < PER_THREAD_Z; z++) {
		int zTemp = globalZ[z] * data.width * data.height;
#pragma unroll
		for (int y = 0; y < PER_THREAD_Y; y++) {
			int yTemp = globalY[y] * data.width;
#pragma unroll
			for (int x = 0; x < PER_THREAD_X; x++) {
//				printf("tidx=%d,tidy=%d,tidz=%d, (x,y,z) = (%d,%d,%d)\n",threadIdx.x,threadIdx.y,threadIdx.z,x,y,z);
//				printf("tidx=%d,tidy=%d,tidz=%d, globalIndex[%d][%d][%d] = %d\n",threadIdx.x,threadIdx.y,threadIdx.z,z,y,x);
//				printf("tidx=%d,tidy=%d,tidz=%d \n",threadIdx.x,threadIdx.y,threadIdx.z);
				globalIndex[z][y][x] = globalX[x] + yTemp + zTemp;
//				printf("After globalIndex assignment\n");
			}
		}
	}

	/*
	 * Copy into shared memory
	 */
//	printf("sizeof(shared) = %d\n",sizeof(shared));
//	printf("sizeof(sharedZ) = %d\n",sizeof(sharedZ));
//	printf("sizeof(sharedY) = %d\n",sizeof(sharedY));
//	printf("sizeof(sharedX) = %d\n",sizeof(sharedX));
//    for (int z = 0; z < PER_THREAD_Z; z++) {
//        for (int y = 0; y < PER_THREAD_Y; y++) {
//            for (int x = 0; x < PER_THREAD_X; x++) {
//                printf("Block(%d, %d, %d) Thread(%d, %d, %d) Iter(%d, %d, %d) Shared(%d, %d, %d) Global(%d, %d, %d) GI(%d)\n", blockCol, blockRow, blockDep, threadCol, threadRow, threadDep, z, y, x, sharedX[x], sharedY[y], sharedZ[z], globalX[x], globalY[y], globalZ[z], globalIndex[z][y][x]);
//            }
//        }
//    }
#pragma unroll
	for (int z = 0; z < PER_THREAD_Z; z++) {
#pragma unroll
		for (int y = 0; y < PER_THREAD_Y; y++) {
#pragma unroll
			for (int x = 0; x < PER_THREAD_X; x++) {
//				if(threadIdx.x == 0) {printf("here 7\n");
//					printf("sharedZ[%d] = %d, sharedY[%d] = %d, sharedX[%d] = %d\n",z,sharedZ[z],y,sharedY[y],x,sharedX[x]); 
//				}
		//		printf("(%d,%d,%d): shared[%d][%d][%d] = %d \n", threadIdx.x,threadIdx.y,threadIdx.z,sharedZ[z],sharedZ[z],sharedZ[z],shared[sharedZ[z]][sharedY[y]][sharedX[x]]);
	//				printf("sharedZ[%d] = %d, sharedY[%d] = %d, sharedX[%d] = %d\n",z,sharedZ[z],y,sharedY[y],x,sharedX[x]); 
				shared[sharedZ[z]][sharedY[y]][sharedX[x]] =
						data.elements[globalIndex[z][y][x]];
			}
		}
	}

	// Copy below-block dependencies into shared memory
	if (threadRow == 0 && blockRow > 0) {
#pragma unroll
		for (int z = 0; z < PER_THREAD_Z; z++) {
#pragma unroll
			for (int x = 0; x < PER_THREAD_X; x++) {
				shared[sharedZ[z]][0][sharedX[x]] =
						data.elements[globalIndex[z][0][x] - data.width];
			}
		}
	}

//	printf("here 8\n");
	// Copy above-block dependencies into shared memory
	if (threadRow == blockDim.y - 1
			&& (blockRow + 1) * TILE_HEIGHT < data.height - 1) {
#pragma unroll
		for (int z = 0; z < PER_THREAD_Z; z++) {
#pragma unroll
			for (int x = 0; x < PER_THREAD_X; x++) {
				shared[sharedZ[z]][TILE_HEIGHT + 1][sharedX[x]] =
						data.elements[globalIndex[z][PER_THREAD_Y - 1][x]
								+ data.width];
			}
		}
	}

//	printf("here 9\n");
	// Copy left-of-block dependencies into shared memory
	if (threadCol == 0 && blockCol > 0) {
#pragma unroll
		for (int z = 0; z < PER_THREAD_Z; z++) {
#pragma unroll
			for (int y = 0; y < PER_THREAD_Y; y++) {
				shared[sharedZ[z]][sharedY[y]][0] =
						data.elements[globalIndex[z][y][0] - 1];
			}
		}
	}

//	printf("here 10\n");

	// Copy right-of-block dependencies into shared memory
	if (threadCol == blockDim.x - 1
			&& (blockCol + 1) * TILE_WIDTH < data.width) {
#pragma unroll
		for (int z = 0; z < PER_THREAD_Z; z++) {
#pragma unroll
			for (int y = 0; y < PER_THREAD_Y; y++) {
				shared[sharedZ[z]][sharedY[y]][TILE_WIDTH + 1] =
						data.elements[globalIndex[z][y][PER_THREAD_X - 1] + 1];
			}
		}
	}

//	printf("here 11\n");
	// Copy in-front-of-block dependencies into shared memory
	if (threadDep == 0 && blockDep > 0) {
#pragma unroll
		for (int y = 0; y < PER_THREAD_Y; y++) {
#pragma unroll
			for (int x = 0; x < PER_THREAD_X; x++) {
				shared[0][sharedY[y]][sharedX[x]] =
						data.elements[globalIndex[0][y][x]
								- data.width * data.height];
			}
		}
	}

//	printf("here 12\n");
	// Copy behind-block dependencies into shared memory
	if (threadDep == blockDim.z - 1
			&& (blockDep + 1) * TILE_DEPTH < data.depth) {
#pragma unroll
		for (int y = 0; y < PER_THREAD_Y; y++) {
#pragma unroll
			for (int x = 0; x < PER_THREAD_X; x++) {
				shared[TILE_DEPTH + 1][sharedY[y]][sharedX[x]] =
						data.elements[globalIndex[PER_THREAD_Z - 1][y][x]
								+ data.width * data.height];
			}
		}
	}

//	printf("here 13\n");
	__syncthreads();

	/*
	 * Calculate Values
	 */
	for (int z = 0; z < PER_THREAD_Z; z++) {
		int globZ = globalZ[z];
		int sharZ = sharedZ[z];
#pragma unroll
		for (int y = 0; y < PER_THREAD_Y; y++) {
			int globY = globalY[y];
			int sharY = sharedY[y];
#pragma unroll
			for (int x = 0; x < PER_THREAD_X; x++) {
				int globX = globalX[x];
				int sharX = sharedX[x];

				if (globX > 0 && globX < data.width - 1 && globY > 0
						&& globY < data.height - 1 && globZ > 0
						&& globZ < data.depth - 1) {
					// Calculate new value
					local[z][y][x] = (shared[sharZ][sharY][sharX]
							+ shared[sharZ][sharY][sharX - 1]
							+ shared[sharZ][sharY][sharX + 1]
							+ shared[sharZ][sharY - 1][sharX]
							+ shared[sharZ][sharY + 1][sharX]
							+ shared[sharZ - 1][sharY][sharX]
							+ shared[sharZ + 1][sharY][sharX]) / 7;
				} else if (globX == 0 || globX == data.width - 1 || globY == 0
						|| globY == data.height - 1 || globZ == 0
						|| globZ == data.depth - 1) {
					// On the edge
					local[z][y][x] = shared[sharZ][sharY][sharX];
				} else {
					// Beyond the edge, shouldn't ever hit this unless we messed something up
				}
			}
		}
	}

//	printf("here 14\n");
	__syncthreads();

#pragma unroll
	for (int z = 0; z < PER_THREAD_Z; z++) {
#pragma unroll
		for (int y = 0; y < PER_THREAD_Y; y++) {
#pragma unroll
			for (int x = 0; x < PER_THREAD_X; x++) {
				result.elements[globalIndex[z][y][x]] = local[z][y][x];
			}
		}
	}
	//printf("here 2\n");
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


	HANDLE_ERROR(cudaMalloc((void ** ) &deviceA.elements, sizeA));
	if (copyToDevice) {
		HANDLE_ERROR(
				cudaMemcpy(deviceA.elements, A.elements, sizeA,
						cudaMemcpyHostToDevice));
	}

	return deviceA;
}

void callKernel(Args args, Matrix A, Matrix B) {
	Matrix deviceA, deviceB;

	deviceA = initialize_device(A, true);
	deviceB = initialize_device(B, false);

//	size_t size = 5000;
//	cudaDeviceSetLimit(cudaLimitPrintfFifoSize,size);
	dim3 blocks(max(args.size / TILE_WIDTH, 1), max(args.size / TILE_HEIGHT, 1),
 args.size/TILE_DEPTH);
	dim3 threads(TILE_WIDTH, TILE_HEIGHT, 1);
	for (int t = 0; t < args.iterations; t++) {
		cached_plane_shared_mem<<<blocks, threads>>>(deviceA, deviceB);
        checkCUDAError("cached_plane_shared_mem", true);
		swap(deviceA, deviceB);
	}
  //  printf("sizeof(deviceA) = %d\n", sizeof(deviceA));
   // printf("sizeof(deviceB) = %d\n", sizeof(deviceB));
    //printf("sizeof(B) = %d\n", sizeof(B));
	HANDLE_ERROR(
			cudaMemcpy(B.elements, deviceA.elements,
					A.width * A.height * A.depth * sizeof(float),
					cudaMemcpyDeviceToHost));
}

// Data output
void print_data(float *data, int size, int dimensions) {
	if (size > 20) {
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
    float runtime;
    struct timeval start, end;
	Matrix A, B;
	A = initialize_matrix(args.dimensions, args.size, args.size, args.size);
	B = initialize_matrix(args.dimensions, args.size, args.size, args.size);

//	atexit(cleanupCuda);

	//if (args.debug) { print_data(data, args.size, args.dimensions); }
    gettimeofday( &start, NULL );
	callKernel(args, A, B);
    gettimeofday( &end, NULL );
    runtime = ( ( end.tv_sec  - start.tv_sec ) * 1000.0 ) + ( ( end.tv_usec - start.tv_usec ) / 1000.0 );
    printf( "Processing Time: %4.4f milliseconds\n", runtime );
	if (args.debug) {
		print_data(B.elements, args.size, args.dimensions);
	}
}
