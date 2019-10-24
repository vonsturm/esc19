#include <thread>
#include <iostream>
#include <chrono>
#include <mutex>
#include <vector>

int main()
{
  std::mutex myMutex;

  auto f = [&myMutex](int i){
    std::lock_guard<std::mutex> myLock(myMutex);
    std::cout << "Hello world from thread " << i << std::endl;
  };

  auto start = std::chrono::system_clock::now();

  // spawn a thread
  int N = 4;
  std::vector<std::thread> v;
  for(int i=0; i<N; i++) v.emplace_back(f,i);

  // delete the thread
  for(auto& t : v) t.join();

  auto stop = std::chrono::system_clock::now();
  std::chrono::duration<double> d = stop - start;
  std::cout << d.count() << " seconds" << std::endl;

  return 0;
}
