#include <thread>
#include <atomic>
#include <tbb/tbb.h>
#include <iostream>
#include <execution>

template<class T>
void dump_vector(std::vector<T> v)
{
  std::cout << "{";
  for(auto vi : v)
    std::cout << " " << vi;
  std::cout << " }" << std::endl;
}

int main(int argc, char * argv[])
{
  if(argc < 3)
  {
    std::cout << "Not enough arguments : " << argc << std::endl;
    return 1;
  }

  unsigned int nbins = atoi(argv[1]);
  float limit_l      = atof(argv[2]);
  float limit_r      = atof(argv[3]);
  float step = (limit_r-limit_l)/nbins;

  // integrate in parallel
  std::atomic<float> integral(0);

  std::vector<float> v(nbins);
  int c = 0;
  for( auto & vi : v ) vi = step*c++;

  dump_vector(v);
  std::for_each(std::execution::par, v.begin(), v.end(), [](float &val){val*=2;});
  dump_vector(v);

  double sum = 0;

  tbb::parallel_for(
    tbb::blocked_range<int>(0,nbins),
    [&](const tbb::blocked_range<int>& range) { for(int i = range.begin(); i < range.end(); ++i){ sum += v[i]*step;}  }
  );

  std::cout << "Integral : " << sum << std::endl;
}
