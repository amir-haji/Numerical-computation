#include <stdio.h>
#include <chrono>  // chrono::system_clock
#include <stdlib.h>
#include <iostream>

using namespace std;
using namespace chrono;


__global__
void compute_derivatives(int *n, float *h, float *x, float *y, float *dev) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i > 1 && i < *n - 2) {
        dev[i] = (y[i - 2] - 8 * y[i - 1] + 8 * y[i + 1] - y[i + 2]) / (12 * (*h));
    } else {
        dev[i] = 0;
    }
}


float f(float x) {
    return x * x;
}

int main() {
    int h_n = 10;
    float h_h = 0.1;
    float h_start = 1;
    float h_x[h_n];
    float h_y[h_n];
    float h_dev[h_n];

    for(int i = 0; i < h_n; i++) {
        h_x[i] = h_start + h_h * i;
        h_y[i] = f(h_x[i]);
    }

    int* d_n;
    float* d_x;
    float* d_y;
    float* d_h;
    float* d_dev;

    cudaMalloc(&d_n, sizeof(int));
    cudaMalloc(&d_h, sizeof(float));
    cudaMalloc(&d_x, h_n  * sizeof(float));
    cudaMalloc(&d_y, h_n  * sizeof(float));
    cudaMalloc(&d_dev, h_n  * sizeof(float));

    cudaMemcpy(d_n, &h_n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, &h_h, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, &h_x, h_n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &h_y, h_n * sizeof(float), cudaMemcpyHostToDevice);

    compute_derivatives<<<1, h_n>>>(d_n, d_h, d_x, d_y, d_dev);

    cudaMemcpy(&h_dev, d_dev, h_n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaError_t e = cudaPeekAtLastError();
    const char* e_str = cudaGetErrorString(e);
    cout << "the error is " << e_str << endl;

    for(int i = 0; i < h_n; i++) {
        cout << h_x[i] << " ";
    }
    cout << endl;

    for(int i = 0; i < h_n; i++) {
        cout << h_y[i] << " ";
    }
    cout << endl;

    for(int i = 0; i < h_n; i++) {
        cout << h_dev[i] << " ";
    }
    cout << endl;
}