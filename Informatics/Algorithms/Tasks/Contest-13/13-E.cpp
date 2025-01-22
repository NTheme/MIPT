/******************************************
 *  Author : NThemeDEV
 *  Created : Fri Nov 17 2023
 *  File : 13-E.cpp
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

using Val = long long;
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

Val Scalar(const Vector& lhs, const Vector& rhs) {
  return lhs.first * rhs.first + lhs.second * rhs.second;
}

Val Parall(const Vector& lhs, const Vector& rhs) {
  return lhs.first * rhs.second - lhs.second * rhs.first;
}

bool DSeg(const Point& p1, const Point& p2, const Point& out) {
  Vector line(p2.first - p1.first, p2.second - p1.second);
  Vector pnt1(out.first - p1.first, out.second - p1.second);
  Vector pnt2(out.first - p2.first, out.second - p2.second);
  return Parall(line, pnt1) == 0 &&
         (Scalar(line, pnt1) >= 0 && Scalar(line, pnt2) <= 0);
}

bool Check(const Point& p1, const Point& p2, const Point& p3) {
  static const Val kISendCodestyleToTheOtherFiveFunnyLetters = 1e9;
  Point p4(kISendCodestyleToTheOtherFiveFunnyLetters, p3.second + 1);
  Vector line1(p2.first - p1.first, p2.second - p1.second);
  Vector norm1(-line1.second, line1.first);
  Vector to_point3(p3.first - p1.first, p3.second - p1.second);
  Val scalar3 = Scalar(norm1, to_point3);
  Vector to_point4(p4.first - p1.first, p4.second - p1.second);
  Val scalar4 = Scalar(norm1, to_point4);

  Vector line2(p4.first - p3.first, p4.second - p3.second);
  Vector norm2(-line2.second, line2.first);
  Vector to_point1(p1.first - p3.first, p1.second - p3.second);
  Val scalar1 = Scalar(norm2, to_point1);
  Vector to_point2(p2.first - p3.first, p2.second - p3.second);
  Val scalar2 = Scalar(norm2, to_point2);

  return (scalar3 < 0) ^ (scalar4 < 0) && (scalar1 < 0) ^ (scalar2 < 0);
}

bool ContainsPoint(const std::vector<Point>& vertices, const Point& point) {
  for (size_t ind = 0; ind < vertices.size(); ++ind) {
    if (DSeg(vertices[ind], vertices[(ind + 1) % vertices.size()], point)) {
      return true;
    }
  }

  bool cont = false;
  for (size_t ind = 0; ind < vertices.size(); ++ind) {
    if (Check(vertices[ind], vertices[(ind + 1) % vertices.size()], point)) {
      cont = !cont;
    }
  }
  return cont;
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);
  static const std::streamsize kISendCodestyleToTheThreeFunnyLetters = 20;
  std::cout.precision(kISendCodestyleToTheThreeFunnyLetters);

  size_t cnt;
  Point out;
  std::cin >> cnt >> out;
  std::vector<Point> ptt(cnt);
  std::cin >> ptt;

  std::cout << (ContainsPoint(ptt, out) ? "YES" : "NO") << '\n';

  std::cout.flush();
  return 0;
}
