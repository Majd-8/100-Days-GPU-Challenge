#include<iostream>
#include<cuda_runtime.h>
#include<time.h>
#include<stdlib.h>

__global__ void matrixMultiplicationKernel2(int *A, int *B, int *C, int A_height, int A_width, int B_height, int B_width)
{
    if(A_width == B_height)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        // printf("row: %d\t, Column: %d\n", row, col);
        // printf("B_width = %d\n", B_width);
        
        if(row < A_height && col == 0)
        {
            for(int i = 0; i < B_width; i++)
            {
                int currCol = col + i;
                //printf("currCol: %d\n", currCol);
                int sum = 0;
                for(int j = 0; j < A_width; j++)
                {
                    //printf("A[%d] * B[%d] = %d * %d\n",(row * A_width + j), (j * B_width + currCol), A[row * A_width + j], B[j * B_width + currCol]);
                    sum += A[row * A_width + j] * B[j * B_width + currCol];
                    //printf("sum: %d\n", sum);
                }
                //printf("C[%d] = %d\n", row*B_width + i, sum);
                C[row*B_width + i] = sum;
            }
        }
    }
}

int main()
{
    int A_height = 2;
    int A_width  = 3;
    int sizeA = A_height * A_width;

    int B_height = 3;
    int B_width  = 4;
    int sizeB = B_height * B_width;

    int sizeC = A_height * B_width;

    int A_MemSpace = A_height * A_width * sizeof(int);
    int B_MemSpace = B_height * B_width * sizeof(int);
    int C_MemSpace = A_height * B_width * sizeof(int);

    int A_h[6];
    int B_h[12];
    int C_h[8];

    for(int i = 0; i < sizeA; i++)
    {
        A_h[i] = rand() % 11;
    }

    for(int i = 0; i < sizeB; i++)
    {
        B_h[i] = rand() % 11;
    }

    printf("A = \n");
    for(int i = 0; i < A_height; i++)
    {
        for(int j = 0; j < A_width; j++)
        {
            printf("%d\t",A_h[i * A_width + j]);
        }
        printf("\n");
    }

    printf("B = \n");
    for(int i = 0; i < B_height; i++)
    {
        for(int j = 0; j < B_width; j++)
        {
            printf("%d\t",B_h[i * B_width + j]);
        }
        printf("\n");
    }

    int *A_d;
    int *B_d;
    int *C_d;
    cudaMalloc((void **)&A_d, A_MemSpace);
    cudaMalloc((void **)&B_d, B_MemSpace);
    cudaMalloc((void **)&C_d, C_MemSpace);
    
    cudaMemcpy(A_d, A_h, A_MemSpace, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, B_MemSpace, cudaMemcpyHostToDevice);

    dim3 dimGrid(2, 1, 1);
    dim3 dimBlock(2, 2, 1);

    matrixMultiplicationKernel2<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, A_height, A_width, B_height, B_width);

    cudaMemcpy(C_h, C_d, C_MemSpace, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    printf("C= \n");
    for(int i = 0; i < A_height; i++)
    {
        for(int j = 0; j < B_width; j++)
        {
            //printf("C[%d] = %d\t", i*B_width+j, C_h[i * B_width + j]);
            printf("%d\t",C_h[i * B_width + j]);
        }
        printf("\n");
    }

}