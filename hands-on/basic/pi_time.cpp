#include <iostream>
#include <chrono>
#include <cassert>
#include <utility>
#include <cstdlib>

using Duration = std::chrono::duration<float>;

constexpr double pi(int n)
{
  assert(n > 0);

  auto const step = 1. / n;
  auto sum = 0.;
  for (int i = 0; i != n; ++i) {
    auto x = (i + 0.5) * step;
    sum += 4. / (1. + x * x);
  }

  return step * sum;
}

int main(int argc, char* argv[])
{
//  int const n = (argc > 1) ? std::atoi(argv[1]) : 10;
  int const n = 1000;

  auto const start = std::chrono::high_resolution_clock::now();
  auto constexpr value = pi(n);
  auto const end = std::chrono::high_resolution_clock::now();

  Duration time = end - start;

  std::cout << "pi = " << value
            << " for " << n << " iterations"
            << " in " << time.count() << " s\n";
}
