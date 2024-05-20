/******************************************
 *  Author : NThemeDEV
 *  Created : Fri Nov 10 2023
 *  File : 12-И.cpp
 ******************************************/

/*
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")
*/

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

template <typename TypeFirst, typename TypeSecond>
std::istream& operator>>(std::istream& inp,
                         std::pair<TypeFirst, TypeSecond>& pair) {
  inp >> pair.first >> pair.second;
  return inp;
}

template <typename TypeFirst, typename TypeSecond>
std::ostream& operator<<(std::ostream& out,
                         const std::pair<TypeFirst, TypeSecond>& pair) {
  out << pair.first << ' ' << pair.second;
  return out;
}

template <typename Type>
std::istream& operator>>(std::istream& inp, std::vector<Type>& array) {
  for (auto& elem : array) {
    inp >> elem;
  }
  return inp;
}

template <typename Type>
std::ostream& operator<<(std::ostream& out, const std::vector<Type>& array) {
  for (const auto& elem : array) {
    out << elem << ' ';
  }
  return out;
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);

  uint32_t nn;
  std::cin >> nn;

  uint32_t ans = 0;
  std::vector<bool> less(nn + 1);
  for (uint32_t div = 2; div < std::sqrt(nn) + 1; ++div) {
    if (less[div]) {
      continue;
    }
    less[div] = true;
    for (uint32_t cur = div * 2; cur < less.size(); cur += div) {
      if (!less[cur]) {
        ans += div;
        less[cur] = true;
      }
    }
  }

  std::cout << ans;

  std::cout.flush();
  return 0;
}
