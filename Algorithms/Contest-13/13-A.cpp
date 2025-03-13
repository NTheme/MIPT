/******************************************
 *  Author : NThemeDEV
 *  Created : Wed Nov 15 2023
 *  File : 13-A.cpp
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

template <typename Type>
Type Length(const std::vector<Type>& vec)
  requires std::is_arithmetic_v<Type>
{
  return std::sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
}

template <typename Type>
std::vector<Type> Sum(const std::vector<Type>& lhs,
                      const std::vector<Type>& rhs)
  requires std::is_arithmetic_v<Type>
{
  return std::vector<Type>({lhs[0] + rhs[0], lhs[1] + rhs[1]});
}

template <typename Type>
Type Scalar(const std::vector<Type>& lhs, const std::vector<Type>& rhs)
  requires std::is_arithmetic_v<Type>
{
  return lhs[0] * rhs[0] + lhs[1] * rhs[1];
}

template <typename Type>
Type Vector(const std::vector<Type>& lhs, const std::vector<Type>& rhs)
  requires std::is_arithmetic_v<Type>
{
  return lhs[0] * rhs[1] - lhs[1] * rhs[0];
}

template <typename Type>
Type Square(const std::vector<Type>& lhs, const std::vector<Type>& rhs)
  requires std::is_arithmetic_v<Type>
{
  return std::abs(Vector(lhs, rhs)) / 2;
}

template <typename Type>
std::vector<Type> ToVector(const std::pair<Type, Type>& p1,
                           const std::pair<Type, Type>& p2)
  requires std::is_arithmetic_v<Type>
{
  return std::vector<Type>({p2.first - p1.first, p2.second - p1.second});
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);
  static const std::streamsize kISendCodestyleToTheThreeFunnyLetters = 20;
  std::cout.precision(kISendCodestyleToTheThreeFunnyLetters);

  std::pair<long double, long double> p1;
  std::pair<long double, long double> p2;
  std::pair<long double, long double> p3;
  std::pair<long double, long double> p4;

  std::cin >> p1 >> p2 >> p3 >> p4;

  auto v1 = ToVector(p1, p2);
  auto v2 = ToVector(p3, p4);

  std::cout << Length(v1) << ' ' << Length(v2) << '\n'
            << Sum(v1, v2) << '\n'
            << Scalar(v1, v2) << ' ' << Vector(v1, v2) << '\n'
            << Square(v1, v2) << '\n';

  std::cout.flush();
  return 0;
}
