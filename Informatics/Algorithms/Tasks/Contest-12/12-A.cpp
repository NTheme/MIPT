/******************************************
 *  Author : NThemeDEV
 *  Created : Wed Nov 08 2023
 *  File : 12-A.cpp
 ******************************************/

/*
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")
*/

#include <algorithm>
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

struct GCDStruct {
  uint64_t gcd;
  std::pair<int64_t, int64_t> var;
};

GCDStruct GCDInternal(const uint64_t& first, const uint64_t& second) {
  GCDStruct ret;
  if (first == 0) {
    ret.var = {0, 1};
    ret.gcd = second;
  } else {
    GCDStruct rec = GCDInternal(second % first, first);
    ret.var = {
        rec.var.second - static_cast<int64_t>(second / first) * rec.var.first,
        rec.var.first};
    ret.gcd = rec.gcd;
  }
  return ret;
}

template <typename TypeFirst, typename TypeSecond>
GCDStruct GCD(const TypeFirst& first, const TypeSecond& second)
  requires std::is_integral_v<TypeFirst> && std::is_integral_v<TypeSecond>
{
  if (first == 0 && second == 0) {
    throw std::logic_error("Infinite GCD!");
  }
  uint64_t first_u = first > 0 ? first : -first;
  uint64_t second_u = second > 0 ? second : -second;
  auto ret = GCDInternal(first_u, second_u);
  ret.var.first = first > 0 ? ret.var.first : -ret.var.first;
  ret.var.second = second > 0 ? ret.var.second : -ret.var.second;
  return ret;
}

template <typename TypeFirst, typename TypeSecond>
GCDStruct GCD(const std::pair<TypeFirst, TypeSecond>& num) {
  return GCD(num.first, num.second);
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

template <typename TypeFirst, typename TypeSecond>
uint64_t ModularInverse(const TypeFirst& num, const TypeSecond& mod) {
  auto str = GCD(num, mod);
  if (str.gcd != 1) {
    throw std::logic_error("No inversed exists!");
  }
  return ToField(str.var.first, mod);
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);

  const long long kMod = 1e9 + 7;
  long long aa;
  long long bb;
  long long cc;
  long long dd;
  std::cin >> aa >> bb >> cc >> dd;

  aa = ((aa * dd + bb * cc) % kMod + kMod);
  bb = (bb * dd) % kMod;

  std::cout << ((ModularInverse(bb, kMod) * aa) % kMod + kMod) % kMod;

  std::cout.flush();
  return 0;
}