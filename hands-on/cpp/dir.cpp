#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <iterator>
#include <dirent.h>

template<typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T> const& c)
{
  os << "{ ";
  std::copy(
      std::begin(c),
      std::end(c),
      std::ostream_iterator<T>{os, " "}
  );
  os << '}';

  return os;
}

std::vector<std::string> entries(DIR*dir)
{
  std::vector<std::string> result;

  // relevant function and data structure are:
  //
  // int  readdir_r(DIR* dirp, struct dirent* entry, struct dirent** result);
  //
  // struct dirent {
  //   // ...
  //   char d_name[256];
  // };
  //
  // dirent entry;
  // for (auto* r = &entry; readdir_r(dir, &entry, &r) == 0 && r; ) {
  //   // here `entry.d_name` is the name of the current entry
  // }
  dirent entry;

  for (auto* r = &entry; readdir_r(dir, &entry, &r) == 0 && r; ) {
    result.push_back(r->d_name);
  }

  return result;
}

int main(int argc, char* argv[])
{
  std::string const name = argc > 1 ? argv[1] : ".";

  // create a smart pointer to a DIR here, with a deleter
  // relevant functions and data structures are
  // DIR* opendir(const char* name);
  // int  closedir(DIR* dirp);
  auto d = std::shared_ptr<DIR>{
    opendir(name.c_str()),
    [](auto d) { closedir(d); std::cout << "I have been destroyed :)" << std::endl; }
  };

  std::vector<std::string> v = entries(d.get());
  std::cout << v << '\n';
}
