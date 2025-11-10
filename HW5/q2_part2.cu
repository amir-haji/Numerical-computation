#include <stdio.h>
#include <chrono>  // chrono::system_clock
#include <stdlib.h>
#include <iostream>

using namespace std;
using namespace chrono;


__global__
void compute_newton(int *j, int *n, float *x, float *mat) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    mat[i * (*n) + *j] = (mat[(i + 1) * (*n) + *j - 1] - mat[(i) * (*n) + *j - 1]) / (x[i + *j] - x[i]);
}

__global__
void compute_terms(float *z, float *x, float *mat, float *terms) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float ans = mat[i];
    for(int j = 0; j < i; j++) {
        ans *= (*z - x[j]);
    }

    terms[i] = ans;
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
    int h_n = 3;
    float h_z = 4;
    float h_x[h_n] = {1, 2, 3};
    float h_y[h_n] = {1, 8, 27};
    float h_mat[h_n * h_n];

    for(int i = 0; i < h_n; i++) {
        for(int j = 0; j < h_n; j++) {
            if(j == 0) {
                h_mat[i * h_n] = h_y[i];
            } else {
                h_mat[i * h_n + j] = 0;
            }
        }
    }


    int* d_n;
    float* d_z;
    float* d_x;
    float* d_mat;


    cudaMalloc(&d_n, sizeof(int));
    cudaMalloc(&d_z, sizeof(float));
    cudaMalloc(&d_x, h_n * sizeof(float));
    cudaMalloc(&d_mat, h_n * h_n * sizeof(float));

    cudaMemcpy(d_n, &h_n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, &h_z, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, &h_x, h_n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat, &h_mat, h_n * h_n * sizeof(float), cudaMemcpyHostToDevice);

    for(int h_j = 1; h_j < h_n; h_j++) {
        int* d_j;
        cudaMalloc(&d_j, sizeof(int));
        cudaMemcpy(d_j, &h_j, sizeof(int), cudaMemcpyHostToDevice);

        compute_newton<<<1, h_n - h_j>>>(d_j, d_n, d_x, d_mat);
    }

    float h_terms[h_n];
    for(int i = 0; i < h_n; i++){ 
        h_terms[i] = 0;
    }

    float* d_terms;
    cudaMalloc(&d_terms, h_n * sizeof(float));
    cudaMemcpy(d_terms, &h_terms, h_n * sizeof(float), cudaMemcpyHostToDevice);

    compute_terms<<<1, h_n>>>(d_z, d_x, d_mat, d_terms);

    
    int block_size = 32; 
    int num_blocks = (int)ceil((h_n + block_size - 1) / block_size);
    int output_size = num_blocks * sizeof(float);
    float* device_output;
    cudaMalloc(&device_output, output_size);

    reduce_sum<<<num_blocks, block_size, block_size * sizeof(float)>>>(d_terms, device_output, d_n);
    reduce_sum<<<1, block_size, block_size * sizeof(float)>>>(device_output, device_output, d_n);

    float *host_output = (float*)malloc(output_size);
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    cudaError_t e = cudaPeekAtLastError();
    const char* e_str = cudaGetErrorString(e);
    cout << "the error is " << e_str << endl;

    cout << host_output[0] << endl;
    


}