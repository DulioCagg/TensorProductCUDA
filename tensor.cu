#include <stdlib.h>
#include <stdio.h>

#define N 2
#define M 3
#define MAX_NUM 10 

// Define Input and Result matrices
int a[N][M], b[N][M], result[N*N][M*M];

// CUDA call, calculate tensor product
// rows, cols = matrix dimensions
// a and b are the input matrices
// result is the result matrix
__global__
void kernelTensorProduct(int rows, int cols, int *a, int *b, int *result)
{
  // Get current thread, identified by its x and y positions
  int i = threadIdx.x;
  int j = threadIdx.y;
  
  // Calculate result matrix column size 
  int totalCols = cols * cols;

  // Iterate over the input matrices
  for (int k = 0; k < rows; k++)
  {
    for (int l = 0; l < cols; l++)
    {
      // Calculate the value of the result matrix
      result[(i * rows + k) * totalCols + (j * cols + l)] = a[i*cols+j] * b[k*cols+l];
    }
  }
}

// Helper function, prints a matrix of size N by M
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

// Helper function, initializes a matrix with random numbers between 0 and MAX_NUM
void initMatrix(int matrix[N][M])
{
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < M; j++)
    {
      matrix[i][j] = rand() % MAX_NUM;
    }
  }
}

// Setup CUDA Environment and calculate tensor product
// rows, cols are the input matrix dimensions
// a and b are the input matrices
// result is the result matrix
void tensorProductDevice(int rows, int cols, int *a, int *b, int *result){
  // Create pointers, used to allocate space for the matrices in the CUDA environment
  int *aD, *bD, *resultD;

  // Calculate input matrix size
  int size = rows * cols * sizeof(int);

  // Calculate output matrix size
  int sizeRes = rows * rows * cols * cols * sizeof(int);

  // Define number of blocks to use with CUDA, 1 block
  dim3 bloques(1,1);

  // Set up number of threads to use with CUDA, N x M threads
  dim3 hilos(N,M);

  // Allocate size for the matrices within the CUDA environment
  cudaMalloc(&aD, size);
  cudaMalloc(&bD, size);
  cudaMalloc(&resultD, sizeRes);

  // Select CUDA Device
  cudaSetDevice(0);

  // Send Input matrices to the CUDA device
  cudaMemcpy(aD, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(bD, b, size, cudaMemcpyHostToDevice);

  // CUDA function call, calculate tensor product
  kernelTensorProduct<<<bloques , hilos>>>(rows, cols, aD, bD, resultD);
  
  // Receive result matrix from the CUDA device
  cudaMemcpy(result, resultD, sizeRes, cudaMemcpyDeviceToHost);
  
  // Free the allocated memory 
  cudaFree(aD);
  cudaFree(bD);
  cudaFree(resultD);
}



int main()
{
  // Initialize input matrices with random numbers
  initMatrix(a);
  initMatrix(b);

  // Call the CUDA setup function
  // This sets up the CUDA environment and calculates the tensor product
  tensorProductDevice(N, M, (int *) a, (int *) b, (int *) result);

  // Print input A
  printf("Matrix A:\n");
  printMatrix(a);

// Print input B
  printf("Matrix B:\n");
  printMatrix(b);

  // Print Results
  printf("Result: R:\n");
  for (int i = 0; i < N * N; i++)
  {
    for (int j = 0; j < M * M; j++)
    {
      printf("%d\t", result[i][j]);
    }
    printf("\n");
  }
  
  // Free memory
  free(a);
  free(b);
  free(result);
}
