#include <cuda_runtime.h>
#include <stdio.h>
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization

// Some kernels assume square blocks
#define BDIMX 16
#define BDIMY 16

__global__ void matrixMultiplication(float *C, float *A, float *B, const int A_rows, const int A_cols, const int B_cols)
{
		/* FIXME */
        // Calculate global thread coordinates
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread is within the matrix bounds
    if(row < A_rows && col < B_cols) {
        float sum = 0.0f;
        
        // Compute matrix multiplication for this element
        for(int k = 0; k < A_cols; k++) {
            sum += A[row * A_cols + k] * B[k * B_cols + col];
        }
        
        // Store the result
        C[row * B_cols + col] = sum;
    }

}

// Shared memory version of matrix multiplication for better performance
__global__ void matrixMultiplicationShared(float *C, float *A, float *B, const int A_rows, const int A_cols, const int B_cols)
{
    // Allocate shared memory for tile
    __shared__ float A_shared[BDIMY][BDIMX+1]; // padding for less bank conflicts
    __shared__ float B_shared[BDIMX][BDIMX+1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    
    float sum = 0.0f;
    
    // Loop over the tiles of A and B matrices
    for (int t = 0; t < (A_cols + BDIMX - 1) / BDIMX; t++) {
        // Load tiles into shared memory
        if (row < A_rows && t * BDIMX + tx < A_cols) {
            A_shared[ty][tx] = A[row * A_cols + t * BDIMX + tx];
        } else {
            A_shared[ty][tx] = 0.0f;
        }
        
        if (t * BDIMX + ty < A_cols && col < B_cols) {
            B_shared[ty][tx] = B[(t * BDIMX + ty) * B_cols + col];
        } else {
            B_shared[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = 0; k < BDIMX; k++) {
            sum += A_shared[ty][k] * B_shared[k][tx];
        }
        
        __syncthreads();
    }
    
    // Store the result
    if (row < A_rows && col < B_cols) {
        C[row * B_cols + col] = sum;
    }
}


#define INDEX(ROW, COL, INNER) ((ROW) * (INNER) + (COL))


void initialData(float *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        in[i] = (float)(rand() & 0xFF) / 10.0f;
    }

    return;
}

void printData(float *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%3.0f ", in[i]);
    }

    printf("\n");
    return;
}

void checkResult(float *hostRef, float *gpuRef, int rows, int cols)
{
    double epsilon = 1.0E-1;
    bool match = 1;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int index = INDEX(i, j, cols);
            if (abs(hostRef[index] - gpuRef[index]) > epsilon) {
                match = 0;
                printf("different on (%d, %d) (offset=%d) element in "
                        "matrix: host %f gpu %f\n", i, j, index,
                        hostRef[index], gpuRef[index]);
                break;
            }
        }
        if (!match) break;
    }

	if (match)
		printf("PASS\n\n");
	else
		printf("FAIL\n\n");
}

void matrixMultiplicationHost(float *C, float *A, float *B, const int A_rows, const int A_cols, const int B_cols)
{
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < A_cols; k++) {
                sum += A[i * A_cols + k] * B[k * B_cols + j];
            }
            C[i * B_cols + j] = sum;
        }
    }
}

int main(int argc, char **argv)
{
    bool iprint = 0;
    bool useSharedMemory = true;  // Set to true to use shared memory version

    // Matrix dimensions
    int A_rows = 1024;    // M
    int A_cols = 512;     // K
    int B_rows = A_cols;  // K
    int B_cols = 256;     // N

    printf("Matrix A: %d x %d\n", A_rows, A_cols);
    printf("Matrix B: %d x %d\n", B_rows, B_cols);
    printf("Matrix C: %d x %d\n", A_rows, B_cols);

    // Calculate memory requirements
    size_t A_bytes = A_rows * A_cols * sizeof(float);
    size_t B_bytes = B_rows * B_cols * sizeof(float);
    size_t C_bytes = A_rows * B_cols * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(A_bytes);
    float *h_B = (float *)malloc(B_bytes);
    float *hostRef = (float *)malloc(C_bytes);
    float *gpuRef = (float *)malloc(C_bytes);

    // Initialize host arrays
    initialData(h_A, A_rows * A_cols);
    initialData(h_B, B_rows * B_cols);

    // Perform matrix multiplication on the host
    matrixMultiplicationHost(hostRef, h_A, h_B, A_rows, A_cols, B_cols);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((float**)&d_A, A_bytes));
    checkCudaErrors(cudaMalloc((float**)&d_B, B_bytes));
    checkCudaErrors(cudaMalloc((float**)&d_C, C_bytes));

    // Copy data from host to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, A_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, B_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_C, 0, C_bytes));

    // Set up execution configuration
    dim3 block(BDIMX, BDIMY);
    dim3 grid((B_cols + block.x - 1) / block.x, (A_rows + block.y - 1) / block.y);

    printf("Launching with grid %d x %d and block %d x %d\n", grid.x, grid.y, block.x, block.y);

    // Launch kernel
    if (useSharedMemory) {
        matrixMultiplicationShared<<<grid, block>>>(d_C, d_A, d_B, A_rows, A_cols, B_cols);
    } else {
        matrixMultiplication<<<grid, block>>>(d_C, d_A, d_B, A_rows, A_cols, B_cols);
    }

    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());
    
    // Copy results back to host
    checkCudaErrors(cudaMemcpy(gpuRef, d_C, C_bytes, cudaMemcpyDeviceToHost));

    if (iprint) {
        printf("First few elements of the result:\n");
        for (int i = 0; i < 10 && i < A_rows; i++) {
            for (int j = 0; j < 10 && j < B_cols; j++) {
                printf("%8.1f ", gpuRef[i * B_cols + j]);
            }
            printf("\n");
        }
    }

    // Check results
    checkResult(hostRef, gpuRef, A_rows, B_cols);

    printf("Matrix multiplication completed\n");

    // Free host and device memory
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    
    return EXIT_SUCCESS;
}
