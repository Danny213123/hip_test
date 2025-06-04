#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

__global__ void addKernel(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1024;
    float *d_a, *d_b, *d_c;
    
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    
    addKernel<<<4, 256>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasDestroy(handle);
    
    cudaFree(d_a);
    cudaFree(d_b); 
    cudaFree(d_c);
    return 0;
}
