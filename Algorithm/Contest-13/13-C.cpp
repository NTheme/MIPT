/******************************************
 *  Author : NThemeDEV
 *  Created : Fri Nov 17 2023
 *  File : -C.cpp
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

Geometry operator*(Geometry vec, const Val& num) {
  vec.first *= num;
  vec.second *= num;
  return vec;
}

Geometry operator+(Geometry lhs, const Vector& rhs) {
  lhs.first += rhs.first;
  lhs.second += rhs.second;
  return lhs;
}

Val Length(const Vector& vec) {
  return std::sqrt(vec.first * vec.first + vec.second * vec.second);
}

Vector Normalize(Vector vec) {
  Val div = Length(vec);
  vec.first /= div;
  vec.second /= div;
  return vec;
}

Val Scalar(const Vector& lhs, const Vector& rhs) {
  return lhs.first * rhs.first + lhs.second * rhs.second;
}

Val Dist(const Point& first, const Point& second) {
  return Length(
      Vector(second.first - first.first, second.second - first.second));
}

Val DSeg(const Point& p1, const Point& p2, const Point& out) {
  Vector line(p2.first - p1.first, p2.second - p1.second);
  Vector norm = Normalize(Vector(-line.second, line.first));
  Vector to_point(out.first - p1.first, out.second - p1.second);
  Val scalar = Scalar(norm, to_point);

  Point base = out + norm * scalar;
  Vector to_base1(base.first - p1.first, base.second - p1.second);
  Vector to_base2(base.first - p2.first, base.second - p2.second);

  if (Scalar(line, to_base1) < 0 || Scalar(line, to_base2) > 0) {
    return std::min(Dist(p1, out), Dist(p2, out));
  }
  return std::abs(scalar);
}

Val Check(const Point& p1, const Point& p2, const Point& p3, const Point& p4) {
  Vector line1(p2.first - p1.first, p2.second - p1.second);
  Vector norm1 = Normalize(Vector(-line1.second, line1.first));
  Vector to_point3(p3.first - p1.first, p3.second - p1.second);
  Val scalar3 = Scalar(norm1, to_point3);
  Vector to_point4(p4.first - p1.first, p4.second - p1.second);
  Val scalar4 = Scalar(norm1, to_point4);

  Vector line2(p4.first - p3.first, p4.second - p3.second);
  Vector norm2 = Normalize(Vector(-line2.second, line2.first));
  Vector to_point1(p1.first - p3.first, p1.second - p3.second);
  Val scalar1 = Scalar(norm2, to_point1);
  Vector to_point2(p2.first - p3.first, p2.second - p3.second);
  Val scalar2 = Scalar(norm2, to_point2);

  if (scalar3 * scalar4 <= 0 && scalar1 * scalar2 <= 0) {
    return 0;
  }
  static const Val kISendCodestyleToTheOtherFiveFunnyLetters = 1e18;
  return kISendCodestyleToTheOtherFiveFunnyLetters;
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);
  static const std::streamsize kISendCodestyleToTheThreeFunnyLetters = 20;
  std::cout.precision(kISendCodestyleToTheThreeFunnyLetters);

  Point p1;
  Point p2;
  Point p3;
  Point p4;
  std::cin >> p1 >> p2 >> p3 >> p4;

  std::cout << std::min(std::min(std::min(DSeg(p1, p2, p3), DSeg(p1, p2, p4)),
                                 std::min(DSeg(p3, p4, p1), DSeg(p3, p4, p2))),
                        Check(p1, p2, p3, p4))
            << '\n';

  std::cout.flush();
  return 0;
}
