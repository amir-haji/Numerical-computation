#include <stdio.h>
#include <chrono>  // chrono::system_clock
#include <stdlib.h>
#include <iostream>

using namespace std;
using namespace chrono;



__global__ void compute_polys(int *n, float *x, float *coeff, float *polys) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float ans = coeff[index];
    for(int i = 0; i < index; i++) {
      ans *= (*x);
    }
    polys[index] = ans;
}

__global__ void reduce_sum(float *input, float *output, int *n) {
    extern __shared__ float shared_data[];

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;


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


int main(){
    int h_n = 3;
    float h_coeff[h_n] = {1, 2, 3};
    float h_x = 2;
    float h_polys[h_n];

    int* d_n;
    float* d_coeff;
    float* d_polys;
    float* d_x;

    cudaMalloc(&d_n, sizeof(int));
    cudaMalloc(&d_coeff, h_n * sizeof(float));
    cudaMalloc(&d_polys, h_n * sizeof(float));
    cudaMalloc(&d_x, sizeof(float));

    cudaMemcpy(d_n, &h_n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coeff, &h_coeff, h_n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_polys, &h_polys, h_n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, &h_x, sizeof(float), cudaMemcpyHostToDevice);

    compute_polys<<<1, h_n>>>(d_n, d_x, d_coeff, d_polys);

    int block_size = 32; 
    int num_blocks = (int)ceil((h_n + block_size - 1) / block_size);
    int output_size = num_blocks * sizeof(float);
    float* device_output;
    cudaMalloc(&device_output, output_size);

    reduce_sum<<<num_blocks, block_size, block_size * sizeof(float)>>>(d_polys, device_output, d_n);
    reduce_sum<<<1, block_size, block_size * sizeof(float)>>>(device_output, device_output, d_n);

    float *host_output = (float*)malloc(output_size);
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaError_t e = cudaPeekAtLastError();
    const char* e_str = cudaGetErrorString(e);
    cout << "the error is " << e_str << endl;

    cout << host_output[0] << endl;

}
