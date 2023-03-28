
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <chrono>


__global__ void oddeven(int* numbers, int size)
{
    int temp;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //numbers[i] = i;
    for (int j = i % 2; j < size; j = j + 2) {
        for (int k = 0; k < size/blockDim.x; k++) {
            i = i + (k * blockDim.x);
            if ((numbers[i] > numbers[i + 1]) && ((i % 2) == 0) && (i < size-1)) {
                temp = numbers[i];
                numbers[i] = numbers[i + 1];
                numbers[i + 1] = temp;
                //numbers[i] = i;
            }
            __syncthreads();
            if ((numbers[i] > numbers[i + 1]) && ((i % 2) != 0) && (i < size - 1)) {
                temp = numbers[i];
                numbers[i] = numbers[i + 1];
                numbers[i + 1] = temp;
                //numbers[i] = i;
            }
            __syncthreads();
        }
    }
}

void print_sort_status(int* numbers, int size)
{
    std::cout << "The input is sorted?: " << (std::is_sorted(&numbers[0], &numbers[size - 1]) == 0 ? "False" : "True") << std::endl;
}

int main()
{
    // Initialize a vector with integers of value 0
    int* numbers;
    int* tmp;
    int* fin;
    int test;
    int k;
    int size = 100000; // Number of elements in the input
    int var = std::min(size-1, 1024);
    //int var = 5;

    srand(time(0));
    // Populate our vector with (pseudo)random numbers
    tmp = (int*)malloc(size * sizeof(int*));
    fin = (int*)malloc(size * sizeof(int*));

    for (k = 0; k < size; k++) {
        test = rand();
        tmp[k] = test;
        //printf("tmp = %d \n", tmp[k]);
    }

    cudaMalloc((void**)&numbers, size * sizeof(int*));
    cudaMemcpy(numbers, tmp, size * sizeof(int*), cudaMemcpyHostToDevice);
    print_sort_status(tmp, size);
    auto start = std::chrono::steady_clock::now();

    int threadsPerBlock(var);
    int numBlocks(size / threadsPerBlock);
    oddeven <<< numBlocks, threadsPerBlock >>> (numbers, size);
    cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();
    cudaMemcpy(fin, numbers, size * sizeof(int*), cudaMemcpyDeviceToHost);
    print_sort_status(fin, size);
    /*for (k = 0; k < size; k++) {
        printf("fin = %d \n", fin[k]);
    }*/
    std::cout << "Elapsed time =  " << std::chrono::duration<double>(end - start).count() << " sec\n";
    cudaFree(numbers);
    return 0;
}