
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


__global__ void oddeven(int* numbers)
{
    int temp;
    if (numbers[threadIdx.x] > numbers[threadIdx.x + 1]) {
        temp = numbers[threadIdx.x];
        numbers[threadIdx.x] = numbers[threadIdx.x + 1];
        numbers[threadIdx.x + 1] = temp;
    }
    __syncthreads();
}

void print_sort_status(int* numbers, int size)
{
    std::cout << "The input is sorted?: " << (std::is_sorted(&numbers[0], &numbers[size - 1]) == 0 ? "False" : "True") << std::endl;
}

int main()
{
    int size = 10; // Number of elements in the input

    // Initialize a vector with integers of value 0
    int* numbers;
    int* tmp;
    int* fin;
    int test;
    int k;
    srand(time(0));
    // Populate our vector with (pseudo)random numbers
    tmp = (int*)malloc(size * sizeof(int*));
    fin = (int*)malloc(size * sizeof(int*));

    for (k = 0; k < size; k++) {
        test = rand() % 100;
        tmp[k] = test;
        printf("rand = %d \n", tmp[k]);
    }


    cudaMalloc((void**)&numbers, size * sizeof(int*));
    cudaMemcpy(numbers, tmp, size * sizeof(int*), cudaMemcpyHostToDevice);

    print_sort_status(tmp, size);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < size; i++) {
        for (int j = i % 2; j < size-1; j = j + 2) { // j = i%2
            oddeven <<< 1, i >>>(numbers);
        }
        cudaDeviceSynchronize();
    }
    
    auto end = std::chrono::steady_clock::now();
    cudaMemcpy(fin, numbers, size*sizeof(int*), cudaMemcpyDeviceToHost);
    print_sort_status(fin, size);
    for (int l = 0; l < size; l++) {
        printf("fin = %d \n", fin[l]);
    }
    std::cout << "Elapsed time =  " << std::chrono::duration<double>(end - start).count() << " sec\n";
    //cudaMemcpy(fin,numbers,size*sizeof(int),cudaMemcpyHostToDevice);
    cudaFree(numbers);
    return 0;
}