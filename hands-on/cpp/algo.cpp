#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <numeric>

std::ostream& operator<<(std::ostream& os, std::vector<int> const& c);
std::vector<int> make_vector(int N);

int main()
{
  // create a vector of N elements, generated randomly
  int const N = 10;
  std::vector<int> v = make_vector(N);
  std::cout << v << '\n';

  // sum all the elements of the vector
  // use std::accumulate
  int sum = std::accumulate(v.begin(),v.end(),0);
  std::cout << sum << '\n';

  // compute the average of the first half and of the second half of the vector
  auto half_it = v.begin()+N/2;
  int avg_first_half  = std::accumulate(v.begin(),half_it,0)  / (N/2);
  int avg_second_half = std::accumulate(half_it,v.end(),0) / (v.size()-N/2);
  std::cout << "half: " << N/2 << " 1st half: " << avg_first_half << " 2nd half: " << avg_second_half << "\n";

  // move the three central elements to the beginning of the vector
  // use std::rotate
  auto mid_it = v.begin()+(v.size()-3)/2;
  auto end_it = mid_it+3;
  std::rotate(v.begin(),mid_it,end_it);
  std::cout << v << '\n';

  // remove duplicate elements
  // use std::sort followed by std::unique/unique_copy
  std::sort(v.begin(),v.end());
  auto last = std::unique(v.begin(), v.end());
  v.erase(last, v.end());
  std::cout << v << '\n';

};

std::ostream& operator<<(std::ostream& os, std::vector<int> const& c)
{
  os << "{ ";
  std::copy(
            std::begin(c),
            std::end(c),
            std::ostream_iterator<int>{os, " "}
            );
  os << '}';

  return os;
}

std::vector<int> make_vector(int N)
{
  std::random_device rd;
  std::mt19937 eng{rd()};

  int const MAX_N = 100;
  std::uniform_int_distribution<int> dist{1, MAX_N};

  std::vector<int> result;
  result.reserve(N);
  std::generate_n(std::back_inserter(result), N, [&]() { return dist(eng); });

  return result;
}
