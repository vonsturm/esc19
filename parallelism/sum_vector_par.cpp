#include <thread>
#include <iostream>
#include <atomic>
#include <vector>
#include <numeric>

int main()
{
  int N = 100000;
  std::vector<int> v(N);

  int c = 0;
  for(auto & vi : v) vi = c++;

  std::atomic<int64_t> sum;
  sum = 0;

  auto f = [&](int start, int end){ int64_t res = std::accumulate(v.begin()+start,v.begin()+end,(int64_t)0); sum += res; return; };

  // spawn the threads
  int num_threads = 4;
  int chunk = N/num_threads;

  std::vector<std::thread> th;

  for(int i=0; i<num_threads; i++)
  {
    if (i==num_threads-1)
      th.emplace_back(f,i*chunk,N);
    else
      th.emplace_back(f,i*chunk,(i+1)*chunk);
  }

  for( auto & t : th ) t.join();

  std::cout << "The sum is " << sum << std::endl;

  return 0;
}
