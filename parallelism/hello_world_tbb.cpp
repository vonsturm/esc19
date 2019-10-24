#include <tbb/tbb.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/mutex.h>
#include <iostream>

// scoped_lock myLock(myMutex)
// spin_mutex

int main()
{
  int n = tbb::task_scheduler_init::default_num_threads();
  int p = 10;
  tbb::task_scheduler_init init(p);

  std::cout << "Hello World! Running " << p << " threads. (default is " << n << ")" << std::endl;

  int N = 1000;
  std::vector<int> x(N,0);

  tbb::parallel_for(
    tbb::blocked_range<int>(0,N,<G>),
    [&](const tbb::blocked_range<int>& range)
    {
      for(int i = range.begin(); i<range.end(); ++i) x[i]++;
    },
    <partitioner>
  );

  return 0;
}
