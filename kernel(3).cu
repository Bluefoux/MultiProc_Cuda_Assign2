/***************************************************************************
 *
 * Sequential version of Gauss-Jordan row reduction
 *
 ***************************************************************************/



    /***************************************************************************
 *
 * Sequential version of Gauss-Jordan row reduction
 *
 ***************************************************************************/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_launch_parameters.h>
#include<cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <cooperative_groups.h>


#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <stdlib.h>


#define MAX_SIZE 4096
#define BLOCKSIZE_x 10
#define BLOCKSIZE_y 10


typedef double matrix[MAX_SIZE][MAX_SIZE];

int	N;		/* matrix size		*/
int	maxnum;		/* max number of element*/
char* Init;		/* matrix init type	*/
int	PRINT;		/* print switch		*/
matrix	A;		/* matrix A		*/
double	b[MAX_SIZE];	/* vector b             */
double	y[MAX_SIZE];	/* vector y             */

/* forward declarations */
void work(void);
void Init_Matrix(void);
void Print_Matrix(void);
void Init_Default(void);
int Read_Options(int, char**);
int iDivUp(int hostPtr, int b);
__global__ void gauswork1(double* numbers, size_t pitcher, double* numy, double* numb, int size, int k);
__global__ void gauswork2(double* numbers, size_t pitcher, double* numy, double* numb, int size, int k);
__global__ void gauswork3(double* numbers, size_t pitcher, double* numy, double* numb, int size, int k);
__global__ void gauswork4(double* numbers, size_t pitcher, double* numy, double* numb, int size, int k);
__global__ void gauswork5(double* numbers, size_t pitcher, double* numy, double* numb, int size, int k);
__global__ void gauswork6(double* numbers, size_t pitcher, double* numy, double* numb, int size, int k);

int
main(int argc, char** argv)
{
    printf("Gauss Jordan\n");
    int i, timestart, timeend, iter;

    Init_Default();		/* Init default values	*/
    Read_Options(argc, argv);	/* Read arguments	*/
    Init_Matrix();		/* Init the matrix	*/
    work();
   if (PRINT == 1)
      Print_Matrix();
}

int iDivUp(int hostPtr, int b) { return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); }

__global__ void gauswork1(double* numbers, size_t pitcher, double* numy, double* numb, int size, int k) {

   
    int    tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int    tidy = blockIdx.y * blockDim.y + threadIdx.y;
   
    if ((tidx < MAX_SIZE) && (tidy < MAX_SIZE))
    {
       
        int i, j, p;

        double* row_a[MAX_SIZE];
        double* row_b;
        double* row_y;

 
        if (tidy == k && tidx > tidy && tidy < size && tidx < size) {
            for (int i = k; i < k + 1; i++) {
                row_a[i] = (double*)((char*)numbers + i * pitcher);
            }

            row_a[k][tidx] = row_a[k][tidx] / row_a[k][tidy];
        }
        __syncthreads();
    }
}
__global__ void gauswork2(double* numbers, size_t pitcher, double* numy, double* numb, int size, int k) {


    int    tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int    tidy = blockIdx.y * blockDim.y + threadIdx.y;

    if ((tidx < MAX_SIZE) && (tidy < MAX_SIZE))
    {

        int i, j, p;

        double* row_a[MAX_SIZE];
        double* row_b;
        double* row_y;


         if (tidy == k && tidx == k)
        {
             for (int i = k; i < k + 1; i++) {
                 row_a[i] = (double*)((char*)numbers + i * pitcher);
             }
             numy[k] = numb[k] / row_a[k][k];
             row_a[k][k] = 1.0;
        }
        __syncthreads();
    }
}

__global__ void gauswork3(double* numbers, size_t pitcher, double* numy, double* numb, int size, int k) {

    int    tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int    tidy = blockIdx.y * blockDim.y + threadIdx.y;

    if ((tidx < MAX_SIZE) && (tidy < MAX_SIZE))
    {

        int i, j, p;

        double* row_a[MAX_SIZE];
        double* row_b;
        double* row_y;

        
        if (tidy > k && tidx > k && tidx < size && tidy < size)
        {
            for (int i = tidy; i < size; i++)
            {
                row_a[i] = (double*)((char*)numbers + i * pitcher);
            }
            row_a[k] = (double*)((char*)numbers + k * pitcher);
            row_a[tidy][tidx] = row_a[tidy][tidx] - row_a[tidy][k] * row_a[k][tidx];
        }
        __syncthreads();
    }
}
__global__ void gauswork4(double* numbers, size_t pitcher, double* numy, double* numb, int size, int k) {

    int    tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int    tidy = blockIdx.y * blockDim.y + threadIdx.y;

    if ((tidx < MAX_SIZE) && (tidy < MAX_SIZE))
    {

        int i, j, p;

        double* row_a[MAX_SIZE];
        double* row_b;
        double* row_y;


        if (tidy > k && tidx == tidy && tidy < size)
        {
            for (int i = tidy; i < tidy+1; i++)
            {
                row_a[i] = (double*)((char*)numbers + i * pitcher);
            }

            numb[tidy] = numb[tidy] - row_a[tidy][k] * numy[k];
            row_a[tidy][k] = 0.0;

        }
        __syncthreads();
    }
}


__global__ void gauswork5(double* numbers, size_t pitcher, double* numy, double* numb, int size, int k) {


    int    tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int    tidy = blockIdx.y * blockDim.y + threadIdx.y;

    if ((tidx < MAX_SIZE) && (tidy < MAX_SIZE))
    {

        int i, j, p;

        double* row_a[MAX_SIZE];
        double* row_b;
        double* row_y;

        if (tidy < k && tidx > k && tidx <=size)
        {
            for (int i = 0; i < k + 1; i++) {
                row_a[i] = (double*)((char*)numbers + i * pitcher);
            }
            row_a[tidy][tidx] = row_a[tidy][tidx] - row_a[tidy][k] * row_a[k][tidx];
        }

    }
}
__global__ void gauswork6(double* numbers, size_t pitcher, double* numy, double* numb, int size, int k) {


    int    tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int    tidy = blockIdx.y * blockDim.y + threadIdx.y;

    if ((tidx < MAX_SIZE) && (tidy < MAX_SIZE))
    {

        int i, j, p;

        double* row_a[MAX_SIZE];
        double* row_b;
        double* row_y;

        if (tidx == tidy && tidy < k)
        {
            for (int i = 0; i < k + 1; i++) {
                row_a[i] = (double*)((char*)numbers + i * pitcher);
            }
            numy[tidy] = numy[tidy] - row_a[tidy][k] * numy[k];
            row_a[tidy][k] = 0.0;
        }
        __syncthreads();
    }
}

void
work(void)
{
    int i;
    int j;
    int k;
    int size = N;
    int var = std::min(size - 1, 1024);
    double* numbers;
    double* numb;
    double* numy;
    
    size_t pitch;

    int kallex = iDivUp(size, BLOCKSIZE_x);
    int kalley = iDivUp(size, BLOCKSIZE_y);
    printf("kallex = %d \n", kallex);
    printf("kalley = %d \n", kalley);

    cudaMallocPitch(&numbers, &pitch, MAX_SIZE * sizeof(double), MAX_SIZE);
    cudaMalloc((void**)&numb, MAX_SIZE * sizeof(double*));
    cudaMalloc((void**)&numy, MAX_SIZE * sizeof(double*));
    //cudaMalloc((void**)&numbers, N * sizeof(double*));
    cudaMemcpy2D(numbers, pitch, A, MAX_SIZE * sizeof(double), MAX_SIZE * sizeof(double), MAX_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(numb, b, MAX_SIZE * sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(numy, y, MAX_SIZE * sizeof(double*), cudaMemcpyHostToDevice);
    
    //int threadsPerBlock(var);
    //int numBlocks(size);

    dim3 gridSize(iDivUp(size, BLOCKSIZE_x), iDivUp(size, BLOCKSIZE_y));
    dim3 blockSize(BLOCKSIZE_y, BLOCKSIZE_x);
    //dim3 gridSize(2, 2);
    //dim3 blockSize(10, 10);
    //printf("kallex = %d \n", kallex);
   // printf("kalley = %d \n", kalley);

    //int threadsPerBlock(var);
    //int numBlocks(size / threadsPerBlock);

    for (k = 0; k < N; k++) {
        int kalle = k;
        gauswork1 << <gridSize, blockSize >> > (numbers, pitch, numy, numb, N, kalle);
        cudaDeviceSynchronize();

        gauswork2 << <gridSize, blockSize >> > (numbers, pitch, numy, numb, N, k);
        cudaDeviceSynchronize();

        gauswork3 << <gridSize, blockSize >> > (numbers, pitch, numy, numb, N, k);
        cudaDeviceSynchronize();
        gauswork4 << <gridSize, blockSize >> > (numbers, pitch, numy, numb, N, k);
        cudaDeviceSynchronize();
        gauswork5 << <gridSize, blockSize >> > (numbers, pitch, numy, numb, N, k);
        cudaDeviceSynchronize();
        gauswork6 << <gridSize, blockSize >> > (numbers, pitch, numy, numb, N, k);
        cudaDeviceSynchronize();
    }

    //gauswork << <numBlocks, threadsPerBlock >> > (numbers, pitch, numy, numb);
    cudaDeviceSynchronize();
    //gauswork << <numBlocks, threadsPerBlock >> > (numbers, pitch);
    //cudaDeviceSynchronize();
    cudaMemcpy2D(A, MAX_SIZE * sizeof(double), numbers, pitch, MAX_SIZE * sizeof(double), MAX_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, numb, size * sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, numy, size * sizeof(double*), cudaMemcpyDeviceToHost);
    cudaFree(numbers);
}

void
Init_Matrix()
{
    int i, j;
    N = 1000;
    printf("\nsize      = %dx%d ", N, N);
    printf("\nmaxnum    = %d \n", maxnum);
    printf("Init	  = %s \n", Init);
    printf("Initializing matrix...");

    if (strcmp(Init, "rand") == 0) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                if (i == j) /* diagonal dominance */
                    A[i][j] = (double)(rand() % maxnum) + 5.0;
                else
                    A[i][j] = (double)(rand() % maxnum) + 1.0;
            }
        }
    }
    if (strcmp(Init, "fast") == 0) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                if (i == j) /* diagonal dominance */
                    A[i][j] = 5.0;
                else
                    A[i][j] = 2.0;
            }
        }
    }

    /* Initialize vectors b and y */
    for (i = 0; i < N; i++) {
        b[i] = 2.0;
        y[i] = 1.0;
    }

    printf("done \n\n");
    //if (PRINT == 1)
      //  Print_Matrix();
}

void
Print_Matrix()
{
    int i, j;

    printf("Matrix A:\n");
    for (i = 0; i < N; i++) {
        printf("[");
        for (j = 0; j < N; j++)
            printf(" %5.2f,", A[i][j]);
        printf("]\n");
    }
    printf("Vector y:\n[");
    for (j = 0; j < N; j++) {
        printf(" %5.2f,", y[j]);
    }
    printf("]\n");
    printf("Vector b:\n[");
    for (j = 0; j < N; j++) {
        printf(" %5.2f,", b[j]);
    }
    printf("]\n");
    printf("\n\n");
}

void
Init_Default()
{
    N = 2048;
    Init = "fast";
    maxnum = 15.0;
    PRINT = 1;
}

int
Read_Options(int argc, char** argv)
{
    char* prog;

    prog = *argv;
    while (++argv, --argc > 0)
        if (**argv == '-')
            switch (*++ * argv) {
            case 'n':
                --argc;
                N = atoi(*++argv);
                break;
            case 'h':
                printf("\nHELP: try sor -u \n\n");
                exit(0);
                break;
            case 'u':
                printf("\nUsage: gaussian [-n problemsize]\n");
                printf("           [-D] show default values \n");
                printf("           [-h] help \n");
                printf("           [-I init_type] fast/rand \n");
                printf("           [-m maxnum] max random no \n");
                printf("           [-P print_switch] 0/1 \n");
                exit(0);
                break;
            case 'D':
                printf("\nDefault:  n         = %d ", N);
                printf("\n          Init      = rand");
                printf("\n          maxnum    = 5 ");
                printf("\n          P         = 0 \n\n");
                exit(0);
                break;
            case 'I':
                --argc;
                Init = *++argv;
                break;
            case 'm':
                --argc;
                maxnum = atoi(*++argv);
                break;
            case 'P':
                --argc;
                PRINT = atoi(*++argv);
                break;
            default:
                printf("%s: ignored option: -%s\n", prog, *argv);
                printf("HELP: try %s -u \n\n", prog);
                break;
            }
}