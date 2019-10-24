#include <threads>
#include <atomic>

int main(int argc, char * argv[])
{
  if(argc < 3)
  {
    return 1;
  }

  unsigned int nbins = argv[1];
  float limit_l      = argv[2];
  float limit_r      = argv[3];
  float step = (limit_r-limit_l)/nbins;

  // define a stupid function f(x) = x;
  auto f = [](float val){ return val; };

  // integrate in parallel
  std::atomic<float> integral(0);

  std::vector<int> v(nbins);
  int c = 0;
  for( auto & vi : v ) vi = step*c++;

  std::transform(std::execute::par, v.begin(), v.end(), f);
}
