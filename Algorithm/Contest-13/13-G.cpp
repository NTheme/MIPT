/******************************************
 *  Author : NThemeDEV
 *  Created : Tue Nov 21 2023
 *  File : 13-П.cpp
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

using Val = long double;
using Geometry = std::pair<Val, Val>;
using Vector = std::pair<Val, Val>;
using Point = std::pair<Val, Val>;

Vector operator-(Vector lhs, const Vector& rhs) {
  lhs.first -= rhs.first;
  lhs.second -= rhs.second;
  return lhs;
}

Val Scalar(const Vector& lhs, const Vector& rhs) {
  return lhs.first * rhs.first + lhs.second * rhs.second;
}

Val Parall(const Vector& lhs, const Vector& rhs) {
  return lhs.first * rhs.second - lhs.second * rhs.first;
}

Val Length(const Vector& vec) {
  return vec.first * vec.first + vec.second * vec.second;
}

std::pair<Val, std::pair<Vector, Vector>> Tang(Vector point, Vector center,
                                               Val radius) {
  Vector vec = point - center;

  const Val kEps = 1e-9;
  if (Length(vec) <= radius * radius + kEps) {
    return {0.0, {point, point}};
  }

  Val tn = std::sqrt(Length(vec) - radius * radius);

  Val coeff = radius * radius / Length(vec);
  Vector vs = {vec.first * coeff, vec.second * coeff};
  coeff = radius * tn / Length(vec);
  Vector vl = {-vec.second * coeff, vec.first * coeff};
  Vector vr = {vec.second * coeff, -vec.first * coeff};

  return {tn,
          {{center.first + vs.first + vl.first,
            center.second + vs.second + vl.second},
           {center.first + vs.first + vr.first,
            center.second + vs.second + vr.second}}};
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);
  static const std::streamsize kISendCodestyleToTheThreeFunnyLetters = 20;
  std::cout.precision(kISendCodestyleToTheThreeFunnyLetters);

  Point point1;
  Point point2;
  Point center;
  Val radius;
  std::cin >> point1.first >> point1.second >> point2.first >> point2.second >>
      center.first >> center.second >> radius;

  auto p1 = Tang(point1, center, radius);
  Vector vl;
  Vector vr;
  if (p1.first == 0) {
    Vector vec = point1 - center;
    vl = {-vec.second, vec.first};
    vr = {vec.second, -vec.first};
  } else {
    vl = p1.second.first - point1;
    vr = p1.second.second - point1;
  }

  if (Parall(vl, point2 - point1) > 0 && Parall(vr, point2 - point1) < 0) {
    auto p2 = Tang(point2, center, radius);
    Val k1 = std::abs(std::atan2(
        Parall(p1.second.first - center, p2.second.second - center),
        Scalar(p1.second.first - center, p2.second.second - center)));
    if (Parall(p1.second.first - center, p2.second.second - center) == 0 &&
        Scalar(p1.second.first - center, p2.second.second - center) < 0) {
      k1 = M_PI;
    }
    Val k2 = std::abs(std::atan2(
        Parall(p1.second.second - center, p2.second.first - center),
        Scalar(p1.second.second - center, p2.second.first - center)));
    if (Parall(p1.second.second - center, p2.second.first - center) == 0 &&
        Scalar(p1.second.second - center, p2.second.first - center) < 0) {
      k2 = M_PI;
    }
    std::cout << p1.first + p2.first + std::min(k1 * radius, k2 * radius);
  } else {
    std::cout << std::sqrt(
        ((point1.first - point2.first) * (point1.first - point2.first) +
         (point1.second - point2.second) * (point1.second - point2.second)));
  }

  std::cout.flush();
  return 0;
}
