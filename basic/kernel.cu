#include <iostream>
#include <cuda_runtime.h>
#include <string>
using namespace std;
__global__ void vector_add(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
		C[i] = A[i] + B[i];
		printf("Thread ID: %d (blockIdx: %d, threadIdx: %d)\n", i, blockIdx.x, threadIdx.x);
	}
}

int main() {
    int N = 1 << 12; 
    size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // 초기화
    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = 2*i;
    }

    // GPU 메모리 할당
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // CPU -> GPU 복사
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	
    // GPU에서 커널 실행
    int threadsPerBlock = 256; // 한 블럭에서 실행되는 thread
    int blocksPerGrid = N/threadsPerBlock ; // 전체 블록 수
	cout << "block per grid : "<<blocksPerGrid << endl;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N); // 블록당 256개 스레드로 실행

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
	}

    // GPU -> CPU 복사
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 결과
	std::cout<< "C[100] = " << h_C[100] << std::endl;

	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}
