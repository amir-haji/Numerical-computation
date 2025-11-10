#include <stdio.h>
#include <chrono>  // chrono::system_clock
#include <stdlib.h>
#include <iostream>

using namespace std;
using namespace chrono;

__global__
void compute_simpsons(double *h, double *y, double *terms) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = 2 * (i + 1);

    terms[i] = (y[j] + 4 * y[j - 1] + y[j - 2]) * (*h) / 3;
}

__global__ void reduce_sum(double *input, double *output, int *n) {
    extern __shared__ double shared_data[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;


    shared_data[tid] = (index < *n) ? input[index] : 0;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}



double f(double x) {
    return x * x;
}

int main() {
    double h_start = 1;
    double h_end = 11;
    // h_n should be an even number
    int h_n = 1000;
    int h_half = (int)(h_n) / 2;
    double h_h = (h_end - h_start) / (double)h_n;
    double h_x[h_n + 1];
    double h_y[h_n + 1];

    for(int i = 0; i <= h_n; i++) {
        h_x[i] = 1 + i * h_h;
        h_y[i] = f(h_x[i]);
    }

    int* d_n;
    int* d_half;
    double* d_h;
    double* d_x;
    double* d_y;
    double* d_terms;

    cudaMalloc(&d_n, sizeof(int));
    cudaMalloc(&d_half, sizeof(int));
    cudaMalloc(&d_h, sizeof(double));
    cudaMalloc(&d_x, (h_n + 1) * sizeof(double));
    cudaMalloc(&d_y, (h_n + 1) * sizeof(double));
    cudaMalloc(&d_terms, h_half * sizeof(double));

    cudaMemcpy(d_n, &h_n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_half, &h_half, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, &h_h, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, &h_x, (h_n + 1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &h_y, (h_n + 1) * sizeof(double), cudaMemcpyHostToDevice);

    compute_simpsons<<<1, h_half>>>(d_h, d_y, d_terms);


    int block_size = 32; 
    int num_blocks = (int)ceil((h_half + block_size - 1) / block_size);
    int output_size = num_blocks * sizeof(double);
    double* device_output;
    cudaMalloc(&device_output, output_size);

    reduce_sum<<<num_blocks, block_size, block_size * sizeof(double)>>>(d_terms, device_output, d_half);
    reduce_sum<<<1, block_size, block_size * sizeof(double)>>>(device_output, device_output, d_half);

    double *host_output = (double*)malloc(output_size);
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    cudaError_t e = cudaPeekAtLastError();
    const char* e_str = cudaGetErrorString(e);
    cout << "the error is " << e_str << endl;

    cout << host_output[0] << endl;


}