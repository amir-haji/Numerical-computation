#include <stdio.h>
#include <chrono>  // chrono::system_clock
#include <stdlib.h>
#include <iostream>

using namespace std;
using namespace chrono;


__global__
void compute_coeff(float *z, int *n, float *x, float *y, float *term) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float ans = 1;
    for(int j = 0; j < *n; j++) {
        if(j != i) {
            ans *= (*z - x[j]) / (x[i] - x[j]);
        }
    }

    term[i] = y[i] * ans;
}


__global__ void reduce_sum(float *input, float *output, int *n) {
    extern __shared__ float shared_data[];

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


int main() {
    int h_n = 4;
    float h_z = 5;
    float h_x[h_n] = {1, 2, 3, 4};
    float h_y[h_n] = {1, 4, 9, 16};
    float h_term[h_n];



    int* d_n;
    float* d_z;
    float* d_x;
    float* d_y;
    float* d_term;


    cudaMalloc(&d_n, sizeof(int));
    cudaMalloc(&d_z, sizeof(float));
    cudaMalloc(&d_x, h_n * sizeof(float));
    cudaMalloc(&d_y, h_n * sizeof(float));
    cudaMalloc(&d_term, h_n * sizeof(float));

    cudaMemcpy(d_n, &h_n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, &h_z, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, &h_x, h_n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &h_y, h_n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_term, &h_term, h_n * sizeof(float), cudaMemcpyHostToDevice);

    compute_coeff<<<1, h_n>>>(d_z, d_n, d_x, d_y, d_term);

    
    int block_size = 32; 
    int num_blocks = (int)ceil((h_n + block_size - 1) / block_size);
    int output_size = num_blocks * sizeof(float);
    float* device_output;
    cudaMalloc(&device_output, output_size);

    reduce_sum<<<num_blocks, block_size, block_size * sizeof(float)>>>(d_term, device_output, d_n);
    reduce_sum<<<1, block_size, block_size * sizeof(float)>>>(device_output, device_output, d_n);

    float *host_output = (float*)malloc(output_size);
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    cudaError_t e = cudaPeekAtLastError();
    const char* e_str = cudaGetErrorString(e);
    cout << "the error is " << e_str << endl;

    cout << host_output[0] << endl;
    


}