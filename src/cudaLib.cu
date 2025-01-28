
#include "cudaLib.cuh"
#include "cpuLib.h"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	// Calculate thread index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Ensure it's within the bounds
	if (idx < size)
		y[idx] = scale * x[idx] + y[idx];
}

int runGpuSaxpy(int vectorSize) {
	uint64_t vectorBytes = vectorSize * sizeof(float);

	std::cout << "Hello GPU Saxpy!\n";

	float * x, * y, * z; // z is for verification
	float * x_d, *y_d; // device vectors
	float scale = 2.0f;

	// Allocate hosts
	x = (float*)malloc(vectorBytes);
	y = (float*)malloc(vectorBytes);
	z = (float*)malloc(vectorBytes);

	if (x == NULL || y == NULL || z == NULL) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}
	
	// Allocate devices
	gpuErrchk(cudaMalloc(&x_d, vectorBytes));
	gpuErrchk(cudaMalloc(&y_d, vectorBytes));

	// Initialize hosts
	vectorInit(x, vectorSize);
	vectorInit(y, vectorSize);

	#ifndef DEBUG_PRINT_DISABLE
		printf("\n Adding vectors: \n");
		printf(" scale = %f\n", scale);
		printf(" x = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", x[i]);
		}
		printf(" ... }\n");
		printf(" y = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", y[i]);
		}
		printf(" ... }\n");
	#endif

	// Copy host to device
	gpuErrchk(cudaMemcpy(x_d, x, vectorBytes, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(y_d, y, vectorBytes, cudaMemcpyHostToDevice));

	// Kernel setup
	int threadsPerBlock = 256;
	int numBlocks = (vectorSize + threadsPerBlock - 1) / threadsPerBlock;

	// Kernel invocation code
	saxpy_gpu<<<numBlocks, threadsPerBlock>>>(x_d, y_d, scale, vectorSize);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// Kernel computation is done, copy device to host
	gpuErrchk(cudaMemcpy(z, y_d, vectorBytes, cudaMemcpyDeviceToHost));

	#ifndef DEBUG_PRINT_DISABLE
		printf(" z = {");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", z[i]);
		}
		printf(" ... }\n");
	#endif

	// Verify computation
	int errorCount = verifyVector(x, y, z, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	// Deallocate hosts
	free(x);
	free(y);
	free(z);

	// Dellocate devices
	gpuErrchk(cudaFree(x_d));
	gpuErrchk(cudaFree(y_d));

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here
	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}
