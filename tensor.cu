#include <stdlib.h>
#include <stdio.h>

#define N 2
#define M 3

int a[N][M], b[N][M], result[N*N][M*M];

__global__
void kernelTensorProduct(int rows, int cols, int *a, int *b, int *result)
{
  int i = threadIdx.x;
  int j = threadIdx.y;
  int totalCols = cols * cols;

  for (int k = 0; k < rows; k++)
  {
    for (int l = 0; l < cols; l++)
    {
      result[(i * rows + k) * totalCols + (j * cols + l)] = a[i*cols+j] * b[k*cols+l];
    }
  }
}

void printMatrix(int matrix[N][M])
{
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < M; j++)
    {
      printf("%d\t", matrix[i][j]);
    }
    printf("\n");
  }
}

void initMatrix(int matrix[N][M])
{
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < M; j++)
    {
      matrix[i][j] = rand() % 10;
    }
  }
}

void tensorProductDevice(int rows, int cols, int *a, int *b, int *result){
  int *aD, *bD, *resultD;
  int size = rows * cols * sizeof(int);
  int sizeRes = rows * rows * cols * cols * sizeof(int);

  dim3 bloques(1,1);
  dim3 hilos(N,M);

  cudaMalloc(&aD, size);
  cudaMalloc(&bD, size);
  cudaMalloc(&resultD, sizeRes);

  cudaSetDevice(0);
  cudaMemcpy(aD, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(bD, b, size, cudaMemcpyHostToDevice);

  kernelTensorProduct<<<bloques , hilos>>>(rows, cols, aD, bD, resultD);
  
  cudaMemcpy(result, resultD, sizeRes, cudaMemcpyDeviceToHost);
  
  cudaFree(aD);
  cudaFree(bD);
  cudaFree(resultD);
}

int main()
{
  initMatrix(a);
  initMatrix(b);

  tensorProductDevice(N, M, (int *) a, (int *) b, (int *) result);

  printf("Matrix A:\n");
  printMatrix(a);

  printf("Matrix B:\n");
  printMatrix(b);

  printf("Result: R:\n");
  for (int i = 0; i < N * N; i++)
  {
    for (int j = 0; j < M * M; j++)
    {
      printf("%d\t", result[i][j]);
    }
    printf("\n");
  }
  
  free(a);
  free(b);
  free(result);
}