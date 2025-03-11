#include <iostream>
#include <cuda_runtime.h>

// void vecAddSeq(float *A_h, float *B_h, float *C_h, int n)
// {
//     for (int i = 0 ; i < n ; ++i)
//     {
//         C_h[i] = A_h[i] + B_h[i];
//         printf("%f \n", C_h[i]);
//     }
// }

//this is the kernel function that is going to be executed by each thread
//global is a keyword to let this function be callable from both device and host but executed on device
__global__ void vecAddKernel(float *A, float *B, float *C, int n)
{
    //index to identify each thread in each block by a unique number from a range rather than a 2-D index (1,2,3...m*n instead of [0][0], [0][1], [0][2]....[m][n] where m is the number of blocks, n is the number of threads in each block)
    //blockIdx: Index of the block (.x because we are only using 1-D *grid*) 
    //blockDim: (x, y, z) dimensions of the block
    //threadIdx: Index of the thread in the block (.x because we are only using 1-D *block*) 
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    //printf("thread #%d, of block #%d, calculating %f + %f", threadIdx.x, blockIdx.x, A[i], B[i]);

    if(i < n)
    {
        C[i] = A[i] + B[i];
    }
}


int main()
{
    // //Sequential approach
    // int n = 5;
    // float a[5] = {1,2,3,4,5};
    // float b[5] = {1,2,3,4,5};
    // float c[5] = {1,2,3,4,5};
    // vecAddSeq(a, b, c, n);
    
    //Parallel approach
    const int N = 50;

    // float *A_h[N], *B_h[N], *C_h[N];
    float *A_h = new float[N];
    float *B_h = new float[N];
    float *C_h = new float[N];

    for(int i  = 0; i < N; ++i)
    {
        A_h[i] = i;
        B_h[i] = i;
    }

    float *A_d, *B_d, *C_d;
    int size = N*sizeof(float);

    //allocate space in device memory
    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    // //to handle errors (such as it there is no more memory space in device memory)
    // cudaError_t err = cudaMalloc((void **)&A_d, size);
    // if(error != cudaSuccess)
    // {
    //     printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    //     exit(EXIT_FAILURE);
    // }

    //copy objects to device memory after allocation
    //Args: destination, source, size, predefined cuda word to specify the direction of copying either Device -> Host (cudaMemcpyDeviceToHost), or vice versa (cudaMemcpyHostToDevice)
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, size, cudaMemcpyHostToDevice);

    //set the number of blocks and threads in a kernel
    //n: number of elements in the vector (data size)
    //ceil(n/256.0): gives the min number of blocks needed to cover the data (256 is the number of threads in a block and its determined by the user or CUDA version), .0 to produce a float from the division
    
    printf("number of blocks: %f\n", ceil(N/2));

    vecAddKernel<<<ceil(N/2.0), 2>>>(A_d, B_d, C_d, N);

    //copy the resulted array back to the host memory
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    //free objects from device memory (return the allocated space to the availabe pool)
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    for(int i = 0; i < N; ++i)
    {
        printf("%f\n", C_h[i]);
    }


    return 0; 
}