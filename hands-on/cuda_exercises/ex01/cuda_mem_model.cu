// includes, system
#include <iostream>
#include <assert.h>
#include <chrono>
#include <iomanip>
#include <cstdlib>

// Here you can set the device ID that was assigned to you
#define MYDEVICE 3

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg);

using Duration = std::chrono::duration<float>;

///////////////////////////////////////////////////////////////////////////////
// Program main
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char * argv[]) 
{
    cudaSetDevice(MYDEVICE);
    // pointer and dimension for host memory
    char* end;
    int64_t dim = std::strtoll(argv[1], &end, 10);
    int64_t dimA = dim;
    float *h_a;

    std::cout << "dim " << dimA << std::endl;

    // pointers for device memory
    float *d_a, *d_b;

    // allocate and initialize host memory
    // Bonus: try using cudaMallocHost in place of malloc
    // it has the same syntax as cudaMalloc, but it enables asynchronous copies
    size_t memSize = dimA*sizeof(float);
    //h_a = (float *) malloc(memSize);
    cudaMallocHost(&h_a, memSize);
    for (int i = 0; i<dimA; ++i)
    {
        h_a[i] = i;
    }

    // Part 1 of 5: allocate device memory
    cudaMalloc(&d_a, memSize);
    cudaMalloc(&d_b, memSize);

    // Part 2 of 5: host to device memory copy
    auto const start1 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_a, h_a, memSize, cudaMemcpyHostToDevice);
    auto const end1 = std::chrono::high_resolution_clock::now();
    Duration d = end1-start1;

    std::cout << std::setprecision(5) << float(memSize)/(8000000000*d.count()) << "GB/s" << std::endl;

    // Part 3 of 5: device to device memory copy
    cudaMemcpy(d_b, d_a, memSize, cudaMemcpyDeviceToDevice);

    // clear host memory
    for (int i=0; i<dimA; ++i )
    {
        h_a[i] = 0.f;
    }

    // Part 4 of 5: device to host copy
    cudaMemcpy(h_a, d_b, memSize, cudaMemcpyDeviceToHost);

    // Check for any CUDA errors
    checkCUDAError("cudaMemcpy calls");

    // verify the data on the host is correct
    for (int i=0; i<dimA; ++i)
    {
        assert(h_a[i] == (float) i);
    }

    // Part 5 of 5: free device memory pointers d_a and d_b
    cudaFree(d_a);
    cudaFree(d_b);

    // Check for any CUDA errors
    checkCUDAError("cudaFree");

    // free host memory pointer h_a
    //free(h_a);
    cudaFreeHost(h_a);

    // If the program makes it this far, then the results are correct and
    // there are no run-time errors.  Good work!
    std::cout << "Correct!" << std::endl;

    return 0;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        std::cerr << "Cuda error: " << msg << " " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }                         
}
