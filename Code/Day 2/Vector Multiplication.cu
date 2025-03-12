#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrixKernel(float *A, float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < N && j < N)
    {
        C[i * N + j] = A[i * N + j] * B[i * N + j];
        printf("A[%d] * B[%d] = %f\n",i * N + j, i * N + j, C[i * N + j]);
    }
}

int main()
{
    const int N = 10;

    float *A_h = new float[N*N];
    float *B_h = new float[N*N];
    float *C_h = new float[N*N];

    for(int i  = 0; i < N; ++i)
    {
        for(int j  = 0; j < N; ++j)
        {
            A_h[i * N + j] = i * N + j;
            B_h[i * N + j] = i * N + j;
            // printf("A[%d] = B[%d] = %d \n",i * N + j, i * N + j, i * N + j);
        }
    }   

    float *A_d, *B_d, *C_d;
    int size = N*N * sizeof(float);

    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, size, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(4, 2, 1);
    dim3 dimGrid(ceil(N / 4.0f), ceil(N / 2.0f), 1);

    matrixKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, N);

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            printf("%f\n", C_h[i * N + j]);
        }
            
    }

    return 0;
}