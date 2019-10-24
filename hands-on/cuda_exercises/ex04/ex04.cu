#include <stdio.h>
#include <iostream>
#include <iomanip>

// Here you can set the device ID that was assigned to you
#define MYDEVICE 3
__global__
void saxpy(unsigned int n, double a, double *x, double *y)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(int argc, char * argv[])
{
  cudaSetDevice(MYDEVICE);

  // 1<<N is the equivalent to 2^N
  unsigned int N = (argc > 1) ? atoi(argv[1]) * (1 << 20) : 20 * (1 << 20);
  size_t memSize = N*sizeof(double);
  double *x, *y, *d_x, *d_y;
  x = (double*)malloc(memSize);
  y = (double*)malloc(memSize);

  cudaMalloc(&d_x, memSize); 
  cudaMalloc(&d_y, memSize);

  for (unsigned int i = 0; i < N; i++) {
    x[i] = 1.0;
    y[i] = 2.0;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMemcpy(d_x, x, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, memSize, cudaMemcpyHostToDevice);

  cudaEventRecord(start);

  int TMAX = 1024;

  dim3 NThreads(TMAX,TMAX,64);

  saxpy<<<(N+TMAX-1)/TMAX, TMAX>>>(N, 2.0, d_x, d_y);

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);

  cudaMemcpy(y, d_y, memSize, cudaMemcpyDeviceToHost);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  double maxError = 0.;
  for (unsigned int i = 0; i < N; i++) {
    maxError = max(maxError, abs(y[i]-4.0));
  }
  
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);

  float throughput = float(memSize*3)/milliseconds; // B/ms = kB/s

  std::cout << "*** REPORT ***" << std::endl;
  std::cout << std::fixed << std::setprecision(0) <<"data : " << memSize*3*0.001*0.001 << " GB ; TP = " << throughput*0.001*0.001 << " GB/s" << std::endl;

}



