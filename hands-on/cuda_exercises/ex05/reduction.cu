#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <chrono>

// Here you can set the device ID that was assigned to you
#define MYDEVICE 3
#define BLOCKSIZE 1024

using Duration = std::chrono::duration<float>;

double random_double(void)
{
  constexpr double RAND_MAX_INV = 1./RAND_MAX;
  return static_cast<double>(rand()) * RAND_MAX_INV;
}


__global__ void block_sum(const double *input,
                          double *per_block_results,
                          const size_t n)
{
  //fill me
  __shared__ double sdata[BLOCKSIZE];

  int g_index = threadIdx.x + blockIdx.x * blockDim.x;
  int s_index = threadIdx.x;

  // copy in shared memory
  sdata[s_index] = (g_index < n) ? input[g_index] : 0.;

  // sync
/*
  __syncthreads();
  if(s_index == 0 && n < BLOCKSIZE)
  {
    printf("n is %d \n{\n", n);
    for(int i = 0; i < BLOCKSIZE; i++) printf("%.3f,",sdata[i]);
    printf("}\n");
  }
*/
  __syncthreads();

  // reduction
  int sdata_size = BLOCKSIZE;
  bool loop = (n==1) ? false : true;

  while(__syncthreads_or(loop))
  {
    int half_size = (sdata_size+1)/2;
    int r_index = s_index + half_size;

    if(r_index >= half_size && r_index < sdata_size)
    {
      sdata[s_index] += sdata[r_index];
    }

    sdata_size = half_size;
    if(sdata_size==1) loop = false;

    __syncthreads();
  }

  if(s_index == 0)
  {
    per_block_results[blockIdx.x] = sdata[0];
//    printf("We are block %d and our result is %.3f\n",blockIdx.x, sdata[0]);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char * argv[])
{
  // create array of 256ki elements
  const int num_elements = (argc > 1) ? 1<<atoi(argv[1]) : 1<<18;
  srand(time(NULL));
  // generate random input on the host
  std::vector<double> h_input(num_elements);
  for(int i = 0; i < num_elements; ++i)
  {
    h_input[i] = random_double();
  }

  auto const start_cpu = std::chrono::high_resolution_clock::now();
  const double host_result = std::accumulate(h_input.begin(), h_input.end(), 0.);
  auto const stop_cpu = std::chrono::high_resolution_clock::now();
  Duration time_cpu = stop_cpu - start_cpu;

  auto const start_gpu = std::chrono::high_resolution_clock::now();

  //move input to device memory
  double *d_input = 0;
  int memSize = num_elements * sizeof(double);
  cudaMalloc(&d_input, memSize);
  cudaMemcpy(d_input, &h_input[0], memSize, cudaMemcpyHostToDevice);

  // Part 1 of 6: allocate the partial sums: How much space does it need?
  double *d_partial_sums_and_total = 0;
  dim3 block_size; block_size.x = BLOCKSIZE;
  dim3 num_blocks; num_blocks.x = (num_elements+block_size.x-1)/block_size.x;
  memSize = num_blocks.x * sizeof(double);
  cudaMalloc(&d_partial_sums_and_total, memSize);

  // Part 1 of 6: launch one kernel to compute, per-block, a partial sum. How much shared memory does it need?
  block_sum<<<num_blocks,block_size>>>(d_input, d_partial_sums_and_total, num_elements);

  // Part 1 of 6: compute the sum of the partial sums
  block_sum<<<1,block_size>>>(d_partial_sums_and_total, d_partial_sums_and_total, (size_t)num_blocks.x);

  // Part 1 of 6: copy the result back to the host
  double device_result = 0.;
  cudaMemcpy(&device_result, d_partial_sums_and_total, sizeof(double), cudaMemcpyDeviceToHost);

  auto const stop_gpu = std::chrono::high_resolution_clock::now();
  Duration time_gpu = stop_gpu - start_gpu;

  std::cout << std::fixed << std::setprecision(5);
  std::cout << "Host sum: "   << host_result   << " duration: " << time_cpu.count() << "s" << std::endl;
  std::cout << "Device sum: " << device_result << " duration: " << time_gpu.count() << "s" << std::endl;

  // Part 1 of 6: deallocate device memory
  cudaFree(d_input);
  cudaFree(d_partial_sums_and_total);

  return 0;
}
