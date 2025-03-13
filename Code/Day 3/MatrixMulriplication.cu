#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrixMultiplicationKernel(float *matrix1, float *matrix2, float *resMatrix, int width)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int column = blockIdx.y * blockDim.y + threadIdx.y;

    if((row < width) && (column < width))
    {
        float intermediateValue = 0;
        for(int i = 0; i < width; i++)
        {
            intermediateValue += matrix1[row*width+i] * matrix2[i*width + column];
        }
        resMatrix[row*width+column] = intermediateValue;
    }
}

int main()
{
    const int N = 5;

    float *A_h = new float[N*N];
    float *B_h = new float[N*N];
    float *C_h = new float[N*N];

    for(int i  = 0; i < N; ++i)
    {
        for(int j  = 0; j < N; ++j)
        {
            A_h[i * N + j] = i * N + j;
            B_h[i * N + j] = 10;
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
    
    dim3 dimBlock(8, 4, 1);
    dim3 dimGrid(ceil(N / 8.0f), ceil(N / 4.0f), 1);

    matrixMultiplicationKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, N);

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            printf("%f\t", C_h[i * N + j]);
        }
        printf("\n");
    }

    return 0;
}