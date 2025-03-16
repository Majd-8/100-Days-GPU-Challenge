#include<iostream>
#include<cuda_runtime.h>

__global__ void imageBlurKernel(float *out, float *in, int height, int width, int blurSize)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < height && col < width)
    {
        int pixels = 0;
        float sum = 0;

        int linearizedIndex = row*width + col;

        for(int i = -blurSize; i < blurSize + 1; i++)
        {
            row = row + i;
            for(int j = -blurSize; j < blurSize + 1; j++)
            {
                col = col + j;
                if(row >= 0 && row < M && col >= 0 && col < N)
                {
                    sum += in[row*width + col];
                    pixels++;
                }
            }
        }
        out[linearizedIndex] = sum / pixels;
    }
}

int main()
{
    int height = 10;
    int width  = 10;
    int blur = 3;
    int size = height*width * sizeof(float);

    float *A_h = new float[height][width];
    float *B_h = new float[height][width];

    srand(time(NULL));

    for(int i = 0; i < height*width; i++)
    {
        A[i] = rand();
    }

    printf("A = \n");
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            printf("%f",A_h[i]);
        }
        printf("\n");
    }

    cudaMalloc((void **)&A, size);
    cudaMalloc((void **)&A, size);

    float *A_d, *B_d;
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(2, 2, 1);
    dim3 dimBlock(5, 5, 1);

    imageBlurKernel<<dimGrid, dimBlock>>(B_d, A_d, height, width, blur);

    cudaMemcpy(B_h, B_d, size, cudaMemcpyDeviceToHost);

    printf("A after blurring = \n");
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            printf("%f",B_d[i]);
        }
        printf("\n");
    }


}