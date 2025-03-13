/******************************************
 *  Author : NThemeDEV
 *  Created : Sat Nov 11 2023
 *  File : 12-D.cpp
 ******************************************/

/*
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")
*/

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>
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

std::unordered_map<uint32_t, uint32_t> values;

template <typename TypeFirst, typename TypeSecond, typename TypeThird>
auto DLog(const TypeFirst& num, const TypeSecond& val, const TypeThird& mod)
  requires std::is_unsigned_v<TypeFirst> && std::is_integral_v<TypeSecond> &&
           std::is_integral_v<TypeThird>
{
  uint32_t ret = mod;
  uint32_t mod_sqr = std::sqrt(mod) + 1;

  uint64_t step = 1;
  for (uint32_t ind = 0; ind < mod_sqr; ++ind) {
    step = (step * num) % mod;
  }

  uint64_t left = step;
  for (uint32_t ind = 1; ind <= mod_sqr; ++ind) {
    if (!values.contains(left)) {
      values[left] = ind;
    }
    left = (left * step) % mod;
  }

  uint64_t right = val;
  for (uint32_t ind = 0; ind <= mod_sqr; ++ind) {
    if (values.contains(right)) {
      uint32_t ans = values[right] * mod_sqr - ind;
      if (ans < mod) {
        ret = std::min(ans, ret);
      }
    }
    right = (right * num) % mod;
  }
  
  values.clear();
  return ret;
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);

  uint32_t mod;
  uint32_t num;
  uint32_t val;
  while (std::cin >> mod >> num >> val) {
    uint32_t ans = DLog(num % mod, val % mod, mod);
    if (ans == mod) {
      std::cout << "no solution\n";
    } else {
      std::cout << ans << '\n';
    }
  }

  std::cout.flush();
  return 0;
}
