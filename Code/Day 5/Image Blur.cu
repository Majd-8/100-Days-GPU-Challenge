#include<iostream>
#include<cuda_runtime.h>
#include<time.h>
#include<stdlib.h>

__global__ void imageBlurKernel(float *out, float *in, int height, int width, int blurSize)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //printf("thread (%d,%d) of block (%d,%d)",threadIdx.y,threadIdx.x,blockIdx.y,blockIdx.x);
    if(row < height && col < width)
    {
        int pixels = 0;
        float sum = 0;
        // printf("row %d:\t", row);
        // printf("col %d:\n", col);

        int linearizedIndex = row*width + col;

        for(int i = -blurSize; i < (blurSize + 1); ++i)
        {
            for(int j = -blurSize; j < (blurSize + 1); ++j)
            {
                int blurRow = row + i;
                int blurCol = col + j;
                // printf("blurRow: %d\t", blurRow);
                // printf("blurCol: %d\n", blurCol);
                if(blurRow >= 0 && blurRow < height && blurCol >= 0 && blurCol < width)
                {
                    sum += in[blurRow*width + blurCol];
                    pixels += 1;
                    // printf("sum: %f\t", sum);
                    // printf("pixels: %f\n", pixels);
                }

            }
        }
        out[linearizedIndex] = sum / pixels;
        // printf("out = %f\n",out[linearizedIndex]);
    }
}

int main()
{
    int height = 10;
    int width  = 10;
    int sizeArray = height*width;
    int blur = 1;
    int size = height * width * sizeof(int);

    float *A_h = new float[sizeArray];
    float *B_h = new float[sizeArray];

    for(int i = 0; i < (height*width); i++)
    {
        A_h[i] = rand() % 101;
    }

    printf("A = \n");
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            printf("%f\t",A_h[i * width + j]);
        }
        printf("\n");
    }

    float *A_d, *B_d;
    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(2, 2, 1);
    dim3 dimBlock(5, 5, 1);

    imageBlurKernel<<<dimGrid, dimBlock>>>(B_d, A_d, height, width, blur);

    cudaMemcpy(B_h, B_d, size, cudaMemcpyDeviceToHost);

    printf("A after blurring = \n");
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            printf("%f\t",B_h[i * width + j]);
        }
        printf("\n");
    }
    cudaFree(A_d);
    cudaFree(B_d);

}