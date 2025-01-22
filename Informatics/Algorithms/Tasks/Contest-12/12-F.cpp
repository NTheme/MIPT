/******************************************
 *  Author : NThemeDEV
 *  Created : Mon Nov 13 2023
 *  File : 12-F.cpp.cpp
 ******************************************/

/*
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")
*/

#include <algorithm>
#include <cmath>
#include <complex>
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

void FFT(std::vector<std::complex<long double>>& polynom, bool straight) {
  if (polynom.size() == 1) {
    return;
  }

  std::vector<std::complex<long double>> lhs(polynom.size() / 2);
  std::vector<std::complex<long double>> rhs(polynom.size() / 2);

  uint64_t ind = 0;
  for (size_t even = 0; even < polynom.size(); even += 2) {
    lhs[ind] = polynom[even];
    rhs[ind++] = polynom[even + 1];
  }

  FFT(lhs, straight);
  FFT(rhs, straight);

  long double angle = 2 * M_PI / polynom.size() * (straight ? 1 : -1);
  std::complex<long double> cmp(1, 0);
  std::complex<long double> step(std::cos(angle), std::sin(angle));
  for (size_t ind = 0; ind < polynom.size() / 2; ++ind) {
    polynom[ind] = lhs[ind] + cmp * rhs[ind];
    polynom[ind + polynom.size() / 2] = lhs[ind] - cmp * rhs[ind];
    if (!straight) {
      polynom[ind] /= 2;
      polynom[ind + polynom.size() / 2] /= 2;
    }
    cmp *= step;
  }
}

std::vector<int64_t> FFTMultiply(const std::vector<int64_t>& lhs,
                                 const std::vector<int64_t>& rhs) {
  std::vector<std::complex<long double>> polynom_lhs(lhs.begin(), lhs.end());
  std::vector<std::complex<long double>> polynom_rhs(rhs.begin(), rhs.end());
  size_t size = 1;
  for (; size < std::max(lhs.size(), rhs.size()); size <<= 1) {
  }
  size <<= 1;
  polynom_lhs.resize(size);
  polynom_rhs.resize(size);

  FFT(polynom_lhs, true);
  FFT(polynom_rhs, true);
  for (size_t ind = 0; ind < size; ++ind) {
    polynom_lhs[ind] *= polynom_rhs[ind];
  }
  FFT(polynom_lhs, false);

  std::vector<int64_t> res(size);
  for (size_t ind = 0; ind < size; ++ind) {
    res[ind] = std::roundl(polynom_lhs[ind].real());
  }
  for (; res.back() == 0; res.pop_back()) {
  }

  return res;
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);

  size_t lhs_size;
  size_t rhs_size;
  std::cin >> lhs_size;
  std::vector<int64_t> lhs(lhs_size + 1);
  std::cin >> lhs >> rhs_size;
  std::vector<int64_t> rhs(rhs_size + 1);
  std::cin >> rhs;
  std::reverse(lhs.begin(), lhs.end());
  std::reverse(rhs.begin(), rhs.end());

  auto mul = FFTMultiply(lhs, rhs);
  std::reverse(mul.begin(), mul.end());
  std::cout << mul.size() - 1 << ' ' << mul;

  std::cout.flush();
  return 0;
}
