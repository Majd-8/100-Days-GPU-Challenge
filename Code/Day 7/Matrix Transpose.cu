#include<iostream>
#include<cuda_runtime.h>
#include<stdlib.h>


__global__ void matrixTransposeKernel(int *in, int *out, int height, int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < height && col < width)
    {
        for(int i = 0; i < width; i++)
        {
            out[i * height + row] = in[row * width + i];
        }
    }
    
}

int main()
{
    const int height_h = 5;
    const int width_h = 5;

    int a_h[height_h*width_h];
    int b_h[height_h*width_h];

    int size = height_h*width_h * sizeof(int);

    for(int i = 0; i < height_h*width_h; i++)
    {
        a_h[i] = i;
    }

    printf("A before transposing: \n");
    for(int i = 0; i < height_h; i++)
    {
        for(int j = 0; j < width_h; j++)
        {
           printf("%d\t", a_h[i * width_h + j]); 
        }
        printf("\n");
    }

    int *a_d, *b_d;
    cudaMalloc((void **) &a_d, size);
    cudaMalloc((void **) &b_d, size);

    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(4, 2 , 1);
    dim3 dimBlock(16, 16, 1);

    matrixTransposeKernel<<<dimGrid, dimBlock>>>(a_d, b_d, width_h, height_h);

    cudaMemcpy(b_h, b_d, size, cudaMemcpyDeviceToHost);

    printf("A after transposing: \n");
    for(int i = 0; i < width_h; i++)
    {
        for(int j = 0; j < height_h; j++)
        {
            printf("%d\t", b_h[i * height_h + j]);
        }
        printf("\n");
    }

    cudaFree(a_d);
    cudaFree(b_d);

    return 0;
}