#include <cuda_runtime.h>
#include <stdio.h>
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization

#define BDIMX 16
#define BDIMY 16
#define NUM_STREAMS 4  

// shared mem version for better optimization
__global__ void matrixMultiplication(float *C, float *A, float *B, const int A_rows, const int A_cols, const int B_cols)
{
    // allocate shared memory for tile
    __shared__ float A_shared[BDIMY][BDIMX+1]; // padding for less bank conflicts
    __shared__ float B_shared[BDIMX][BDIMX+1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    
    float sum = 0.0f;
    
    // loop over the tiles of A and B
    for (int t = 0; t < (A_cols + BDIMX - 1) / BDIMX; t++) {
        // load tiles into shared memory
        if (row < A_rows && t * BDIMX + tx < A_cols) {
            A_shared[ty][tx] = A[row * A_cols + t * BDIMX + tx];
        } 
        else {
            A_shared[ty][tx] = 0.0f;
        }
        
        if (t * BDIMX + ty < A_cols && col < B_cols) {
            B_shared[ty][tx] = B[(t * BDIMX + ty) * B_cols + col];
        } 
        else {
            B_shared[ty][tx] = 0.0f;
        }
        
        __syncthreads(); // ensure all threads have finished loading data in shared mem
        
        // compute partial sum for this tile
        for (int k = 0; k < BDIMX; k++) {
            sum += A_shared[ty][k] * B_shared[k][tx];
        }
        
        __syncthreads(); // ensure all threads have finished using current data
    }
    
    if (row < A_rows && col < B_cols) {
        C[row * B_cols + col] = sum;
    }
}

#define INDEX(ROW, COL, INNER) ((ROW) * (INNER) + (COL))

void initialData(float *in, const int size)
{
    for (int i = 0; i < size; i++)
    {
        in[i] = (float)(rand() & 0xFF) / 10.0f;
    }

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

void cu_matrixMultiply(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols)
{

    int B_rows = A_cols;
    
    printf("Matrix A: %d x %d\n", A_rows, A_cols);
    printf("Matrix B: %d x %d\n", B_rows, B_cols);
    printf("Matrix C: %d x %d\n", A_rows, B_cols);

    size_t A_bytes = A_rows * A_cols * sizeof(float);
    size_t B_bytes = B_rows * B_cols * sizeof(float);
    size_t C_bytes = A_rows * B_cols * sizeof(float);

    // calculate per-stream sizes 
    int rows_per_stream = A_rows / NUM_STREAMS;
    size_t A_bytes_per_stream = rows_per_stream * A_cols * sizeof(float);
    size_t C_bytes_per_stream = rows_per_stream * B_cols * sizeof(float);

    //float* h_A, * h_B;

    // pinned memory for streams
    //checkCudaErrors(cudaHostAlloc((void**)&h_A, A_bytes, cudaHostAllocDefault));
    //checkCudaErrors(cudaHostAlloc((void**)&h_B, B_bytes, cudaHostAllocDefault));
    //checkCudaErrors(cudaHostAlloc((void**)&gpuRef, C_bytes, cudaHostAllocDefault));


    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        checkCudaErrors(cudaStreamCreate(&streams[i]));
    }

    float* d_A, * d_B, * d_C;
    checkCudaErrors(cudaMalloc((float**)&d_A, A_bytes));
    checkCudaErrors(cudaMalloc((float**)&d_B, B_bytes));
    checkCudaErrors(cudaMalloc((float**)&d_C, C_bytes));

    // copy matrix B to device since used by all streams
    checkCudaErrors(cudaMemcpy(d_B, B, B_bytes, cudaMemcpyHostToDevice));

    dim3 block(BDIMX, BDIMY);
    dim3 grid_segment((B_cols + block.x - 1) / block.x, (A_rows + block.y - 1) / block.y);

    for (int i = 0; i < NUM_STREAMS; i++) {
        size_t A_offset = i * rows_per_stream * A_cols;
        size_t C_offset = i * rows_per_stream * B_cols;

        // copy segment of A from host to device
        checkCudaErrors(cudaMemcpyAsync(&d_A[A_offset], &A[A_offset], A_bytes_per_stream, cudaMemcpyHostToDevice, streams[i]));

        // call kernel
        matrixMultiplication <<<grid_segment, block, 0, streams[i] >> > (&d_C[C_offset], &d_A[A_offset], d_B, rows_per_stream, A_cols, B_cols);

        // copy segment of C back to host
        checkCudaErrors(cudaMemcpyAsync(&C[C_offset], &d_C[C_offset], C_bytes_per_stream, cudaMemcpyDeviceToHost, streams[i]));
    }

    // synchronize streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        checkCudaErrors(cudaStreamSynchronize(streams[i]));
    }

    //cudaMemcpyFromSymbol(&out, d_, C_bytes);
    // Free device memory
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    /*checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(gpuRef));
    return out;*/
}

int main(int argc, char **argv)
{
    // matrix dimensions
    int A_rows = 1024;    
    int A_cols = 512;     
    int B_rows = A_cols; // rows and cols need to be same
    int B_cols = 256;     

    printf("Matrix A: %d x %d\n", A_rows, A_cols);
    printf("Matrix B: %d x %d\n", B_rows, B_cols);
    printf("Matrix C: %d x %d\n", A_rows, B_cols);

    size_t A_bytes = A_rows * A_cols * sizeof(float);
    size_t B_bytes = B_rows * B_cols * sizeof(float);
    size_t C_bytes = A_rows * B_cols * sizeof(float);

    // calculate per-stream sizes 
    int rows_per_stream = A_rows / NUM_STREAMS;
    size_t A_bytes_per_stream = rows_per_stream * A_cols * sizeof(float);
    size_t C_bytes_per_stream = rows_per_stream * B_cols * sizeof(float);
    
    float *h_A, *h_B, *hostRef, *gpuRef;
    
    // pinned memory for streams
    checkCudaErrors(cudaHostAlloc((void**)&h_A, A_bytes, cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc((void**)&h_B, B_bytes, cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc((void**)&hostRef, C_bytes, cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc((void**)&gpuRef, C_bytes, cudaHostAllocDefault));

    // Initialize host arrays
    initialData(h_A, A_rows * A_cols);
    initialData(h_B, B_rows * B_cols);

    // host matrix multiplication
    matrixMultiplicationHost(hostRef, h_A, h_B, A_rows, A_cols, B_cols);

    dim3 block(BDIMX, BDIMY);
    dim3 grid_segment((B_cols + block.x - 1) / block.x, (rows_per_stream + block.y - 1) / block.y); // for streams

    // streams for DMA transfers and computation overlap
    printf("Using %d CUDA streams with DMA transfers\n", NUM_STREAMS);
    printf("Rows per stream: %d\n", rows_per_stream);
    printf("Launching with grid %d x %d and block %d x %d per stream\n", grid_segment.x, grid_segment.y, block.x, block.y);

    // create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        checkCudaErrors(cudaStreamCreate(&streams[i]));
    }

    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((float**)&d_A, A_bytes));
    checkCudaErrors(cudaMalloc((float**)&d_B, B_bytes));
    checkCudaErrors(cudaMalloc((float**)&d_C, C_bytes));

    // copy matrix B to device since used by all streams
    checkCudaErrors(cudaMemcpy(d_B, h_B, B_bytes, cudaMemcpyHostToDevice));

    for (int i = 0; i < NUM_STREAMS; i++) {
        size_t A_offset = i * rows_per_stream * A_cols;
        size_t C_offset = i * rows_per_stream * B_cols;
        
        // copy segment of A from host to device
        checkCudaErrors(cudaMemcpyAsync(&d_A[A_offset], &h_A[A_offset], A_bytes_per_stream, cudaMemcpyHostToDevice, streams[i]));
        
        // call kernel
        matrixMultiplication<<<grid_segment, block, 0, streams[i]>>>(&d_C[C_offset], &d_A[A_offset], d_B, rows_per_stream, A_cols, B_cols);

        // copy segment of C back to host
        checkCudaErrors(cudaMemcpyAsync(&gpuRef[C_offset], &d_C[C_offset], C_bytes_per_stream, cudaMemcpyDeviceToHost, streams[i]));
    }

    // synchronize streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        checkCudaErrors(cudaStreamSynchronize(streams[i]));
    }

    // Free device memory
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    checkResult(hostRef, gpuRef, A_rows, B_cols);
    printf("Matrix multiplication completed\n");

    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(hostRef));
    checkCudaErrors(cudaFreeHost(gpuRef));

    
    return EXIT_SUCCESS;
}