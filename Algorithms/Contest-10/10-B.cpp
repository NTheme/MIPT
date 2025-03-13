/*
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

*/

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

using std::pair;
using std::shared_ptr;
using std::string;
using std::vector;

template <typename TypeFirst, typename TypeSecond>
std::istream& operator>>(std::istream& inp, pair<TypeFirst, TypeSecond>& pair) {
  inp >> pair.first >> pair.second;
  return inp;
}
template <typename TypeFirst, typename TypeSecond>
std::ostream& operator<<(std::ostream& out,
                         const pair<TypeFirst, TypeSecond>& pair) {
  out << pair.first << ' ' << pair.second;
  return out;
}

template <typename Type>
std::istream& operator>>(std::istream& inp, vector<Type>& array) {
  for (auto& elem : array) {
    inp >> elem;
  }
  return inp;
}
template <typename Type>
std::ostream& operator<<(std::ostream& out, const vector<Type>& array) {
  for (const auto& elem : array) {
    out << elem << ' ';
  }
  return out;
}

vector<size_t> PrefixFunc(const string& str) {
  vector<size_t> pi(str.size());
  for (size_t ind = 1; ind < str.size(); ++ind) {
    size_t last = pi[ind - 1];
    while (last > 0 && str[ind] != str[last]) {
      last = pi[last - 1];
    }
    if (str[ind] == str[last]) {
      ++last;
    }
    pi[ind] = last;
  }
  return pi;
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);
  // std::cout.precision(20);

  string str;
  std::cin >> str;

  size_t ans = 1;
  for (size_t start = 0; start < str.size(); ++start) {
    auto pi = PrefixFunc(str.substr(start, str.size() - start));
    for (size_t pos = 0; pos < pi.size(); ++pos) {
      if ((pos + 1) % (pos + 1 - pi[pos]) == 0) {
        ans = std::max(ans, (pos + 1) / (pos + 1 - pi[pos]));
      }
    }
  }

  std::cout << ans << '\n';

  std::cout.flush();
  return 0;
}
