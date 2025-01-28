
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
		y[idx] += scale * x[idx];
}

int runGpuSaxpy(int vectorSize) {
	uint64_t vectorBytes = vectorSize * sizeof(float);

	std::cout << "Hello GPU Saxpy!\n";

	float * x, * y, * z;
	float * x_d, *y_d;
	float scale = 2.0f;

	// Allocate hosts
	x = (float*)malloc(vectorBytes);
	y = (float*)malloc(vectorBytes);

	// Z is for verification
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
	int numBlocks = ceil(vectorSize/256.0);

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

	// Deallocate devices
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
	uint64_t hitCount = 0;

	// Calculate thread index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Don't execute thread if index > expected size
	if (idx >= pSumSize) {
		return;
	}

	// Setup RNG
	curandState_t rng;
	curand_init(clock64(), idx, 0, &rng);

	// Main GPU Monte-Carlo Code - Get random values
	for (int i = 0; i < sampleSize; i++) {
		float x = curand_uniform(&rng);
		float y = curand_uniform(&rng);

		// If x^2 + y^2 <= 1.0f, increment hitCount
		if ((x * x + y * y) <= 1.0f) {
			++hitCount;
		}
	}

	// Store hitCount result in each thread in pSums array
	pSums[idx] = hitCount;
	
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	uint64_t pSum = 0;

	// Calculate thread index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// If pSumSize isn't evenly divisible by reduceSize
	if (idx >= (pSumSize + reduceSize - 1) / reduceSize) {
		return;
	}

	uint64_t reduceIdx = idx * reduceSize;
	
	// Accumulate into pSum
	for (uint64_t i = 0; i < reduceSize; ++i) {
		if (reduceIdx + i < pSumSize) {
			pSum += pSums[reduceIdx + i];
		}
	}

	// Store pSum result in each thread in totals array
	totals[idx] = pSum;
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

	uint64_t * totals;
	uint64_t * hits_d, * totals_d;
	uint64_t totalHitCount = 0;
	uint64_t vectorBytes = generateThreadCount * sizeof(uint64_t);
	uint64_t reduceBytes = reduceThreadCount * sizeof(uint64_t);

	// Allocate hosts
	totals = (uint64_t *)malloc(reduceBytes);

	if (totals == NULL) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}

	// Allocate devices
	gpuErrchk(cudaMalloc(&hits_d, vectorBytes));
	gpuErrchk(cudaMalloc(&totals_d, reduceBytes));

	// Kernel setup
	int threadsPerBlock = 256;
	int numBlocks = ceil(generateThreadCount/256.0);

	// Kernel invocation code
	generatePoints<<<numBlocks, threadsPerBlock>>>(hits_d, generateThreadCount, sampleSize);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	reduceCounts<<<numBlocks, threadsPerBlock>>>(hits_d, totals_d, generateThreadCount, reduceSize);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// Kernel computation is done, copy device to host
	gpuErrchk(cudaMemcpy(totals, totals_d, reduceBytes, cudaMemcpyDeviceToHost));

	// Accumulate all totals from different threads
	for (int i = 0; i < reduceThreadCount; ++i) {
		totalHitCount += totals[i];
	}

	approxPi = ((double)totalHitCount / sampleSize) / generateThreadCount;
	approxPi = approxPi * 4.0f;

	return approxPi;
}
