#include<stdio.h>
#include<cuda_runtime.h>

#define kernel_size 3
#define input_size 8

__global__
void convolution2DKernel(float *input, float *output, float *kernel, 
                        int input_height, int input_width, 
                        int kernel_height, int kernel_width)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
   
    if(row < input_height && col < input_width)
    {
        float intermediateValue = 0;

        for(int i = 0; i < kernel_height; i++)
        {
            for(int j = 0; j < kernel_width; j++)
            {
                int alignedRow = row - (kernel_height/2) + i;
                int alignedCol = col - (kernel_width/2) + j;

                if(alignedRow < input_height && alignedRow >= 0 && alignedCol < input_width && alignedCol >= 0)
                {
                    intermediateValue += input[alignedRow * input_width + alignedCol] * kernel[i * kernel_width + j];
                }
            }
        }
        output[row * input_width + col] = intermediateValue;
    }
}

int main()
{
    float input_h[input_size*input_size];
    float output_h[input_size*input_size];
    int inputMemSpace = input_size*input_size * sizeof(float);
    int kernelMemSpace = kernel_size*kernel_size * sizeof(float);

    float kernel_h[kernel_size*kernel_size] = 
    {
        0.0, -1.0, 0.0,
        -1.0, 5.0, -1.0,
        0.0, -1.0, 0.0
    };

    for(int i = 0; i < input_size * input_size; i++)
    {
        input_h[i] = rand()%11;
    }

    printf("Input array: \n");

    for(int i=0; i < input_size; i++)
    {
        for(int j = 0; j < input_size;  j++)
        {
            printf("%.1f \t", input_h[i * input_size + j]);
        }
        printf("\n");
    }

    printf("Kernel: \n");

    for(int i=0; i < kernel_size; i++)
    {
        for(int j = 0; j < kernel_size;  j++)
        {
            printf("%.1f \t", kernel_h[i * kernel_size + j]);
        }
        printf("\n");
    }

    float *input_d, *output_d, *kernel_d;

    cudaMalloc((void **)&input_d, inputMemSpace);
    cudaMalloc((void **)&output_d, inputMemSpace);
    cudaMalloc((void **)&kernel_d, kernelMemSpace);

    cudaMemcpy(input_d, input_h, inputMemSpace, cudaMemcpyHostToDevice);
    cudaMemcpy(output_d, output_h, inputMemSpace, cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_d, kernel_h, kernelMemSpace, cudaMemcpyHostToDevice);

    dim3 dimBlock(2, 2, 1);
    dim3 dimGrid((input_size + dimBlock.x-1)/dimBlock.x,(input_size + dimBlock.y-1)/dimBlock.y, 1);

    convolution2DKernel<<<dimGrid, dimBlock>>>(input_d, output_d, kernel_d, input_size, input_size, kernel_size, kernel_size);

    cudaMemcpy(output_h, output_d, inputMemSpace, cudaMemcpyDeviceToHost);

    printf("Output array: \n");
    for(int i = 0; i < input_size; i++)
    {
        for(int j = 0; j < input_size; j++)
        {
            printf("%.1f \t", output_h[i * input_size + j]);
        }
        printf("\n");
    }
    return 0;

}