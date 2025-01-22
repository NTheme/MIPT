/******************************************
 *  Author : NThemeDEV
 *  Created : Sat Nov 11 2023
 *  File : 12-E.cpp
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

template <typename TypeFirst, typename TypeSecond>
uint64_t ToField(const TypeFirst& num, const TypeSecond& mod)
  requires std::is_integral_v<TypeFirst> && std::is_integral_v<TypeSecond>
{
  if (mod == 0) {
    throw std::logic_error("Division by zero!");
  }
  uint64_t mod_u = mod > 0 ? mod : -mod;
  if (std::is_unsigned_v<TypeFirst>) {
    return num % mod_u;
  }
  return (num + (num > 0 ? 0 : (-num + mod_u - 1) / mod_u * mod_u)) % mod_u;
}

template <typename TypeFirst, typename TypeSecond, typename TypeThird>
uint64_t DPow(TypeFirst num, TypeSecond pow, const TypeThird& mod)
  requires std::is_integral_v<TypeSecond>
{
  uint64_t mod_u = mod > 0 ? mod : -mod;
  uint64_t num_u = ToField(num, mod);

  if (num == 0) {
    return 0;
  }

  uint64_t res = 1;
  while (pow != 0) {
    if (pow % 2 == 1) {
      res = (res * num_u) % mod_u;
      --pow;
    } else {
      num_u = (num_u * num_u) % mod_u;
      pow >>= 1;
    }
  }

  return res;
}

uint64_t DRoot(uint64_t num, uint64_t mod) {
  if (mod == 2 || num == 0) {
    return num;
  }
  if (DPow(num, (mod - 1) / 2, mod) == mod - 1) {
    return mod;
  }

  uint64_t pow = 1;
  uint64_t odd = mod - 1;
  for (; odd % 2 == 0; odd /= 2) {
    pow *= 2;
  }

  uint64_t not_sq = 1;
  for (; DPow(not_sq, (mod - 1) / 2, mod) == 1;
       not_sq = std::rand() % (mod - 1) + 1) {
  }
  not_sq = DPow(not_sq, odd, mod);

  uint64_t u_i = DPow(num, odd, mod);
  uint64_t v_i = DPow(num, (odd + 1) / 2, mod);
  while (u_i != 1) {
    uint64_t ord = 1;
    uint64_t buf = u_i;
    for (; buf != 1; buf = (buf * buf) % mod) {
      ord *= 2;
    }

    u_i = (u_i * DPow(not_sq, pow / ord, mod)) % mod;
    v_i = (v_i * DPow(not_sq, pow / (2 * ord), mod)) % mod;
  }
  return v_i;
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);
  std::srand(time(nullptr));

  uint16_t tt;
  std::cin >> tt;
  while (tt-- > 0) {
    uint64_t num;
    uint64_t mod;
    std::cin >> num >> mod;
    uint64_t ans = DRoot(num, mod);
    if (ans == mod) {
      std::cout << "IMPOSSIBLE";
    } else {
      std::cout << ans << '\n';
    }
  }

  std::cout.flush();
  return 0;
}
