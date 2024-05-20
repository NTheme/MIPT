/******************************************
 *  Author : NThemeDEV
 *  Created : Wed Nov 15 2023
 *  File : 13-B.cpp
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
Val Dist(const Point& first, const Point& second) {
  return Length(
      Vector(second.first - first.first, second.second - first.second));
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


Vector Perpendicular(const Vector& vec) {
  return Normalize(Vector(-vec.second, vec.first));
}

Val DLin(const Point& p1, const Point& p2, const Point& out) {
  Vector line(p2.first - p1.first, p2.second - p1.second);
  Vector norm = Perpendicular(line);
  Vector to_point(out.first - p1.first, out.second - p1.second);
  Val scalar = Scalar(norm, to_point);
  return std::abs(scalar);
}

Val DRow(const Point& p1, const Point& p2, const Point& out) {
  Vector line(p2.first - p1.first, p2.second - p1.second);
  Vector norm = Perpendicular(line);
  Vector to_point(out.first - p1.first, out.second - p1.second);
  Val scalar = Scalar(norm, to_point);

  Point base = out + norm * scalar;
  Vector to_base1(base.first - p1.first, base.second - p1.second);

  if (Scalar(line, to_base1) < 0) {
    return Dist(p1, out);
  }
  return std::abs(scalar);
}

Val DSeg(const Point& p1, const Point& p2, const Point& out) {
  Vector line(p2.first - p1.first, p2.second - p1.second);
  Vector norm = Perpendicular(line);
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

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);
  static const std::streamsize kISendCodestyleToTheThreeFunnyLetters = 20;
  std::cout.precision(kISendCodestyleToTheThreeFunnyLetters);

  Vector p1;
  Vector p2;
  Vector p3;
  std::cin >> p3 >> p1 >> p2;

  std::cout << DLin(p1, p2, p3) << '\n'
            << DRow(p1, p2, p3) << '\n'
            << DSeg(p1, p2, p3) << '\n';

  std::cout.flush();
  return 0;
}
