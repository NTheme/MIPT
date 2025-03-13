/******************************************
 *  Author : NThemeDEV
 *  Created : Fri Nov 17 2023
 *  File : 13-D.cpp
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

using Val = long long;
using Geometry = std::pair<Val, Val>;
using Vector = std::pair<Val, Val>;
using Point = std::pair<Val, Val>;

Val VectorMul(const Vector& lhs, const Vector& rhs) {
  return lhs.first * rhs.second - lhs.second * rhs.first;
}

Vector operator-(Vector lhs, const Vector& rhs) {
  lhs.first -= rhs.first;
  lhs.second -= rhs.second;
  return lhs;
}

bool IsConvex(const std::vector<Point>& vertices) {
  size_t ind_cnt = 0;
  Point first_segment;
  Point second_segment;
  bool orientation = false;
  while (true) {
    first_segment =
        Point(vertices[ind_cnt] -
              vertices[(ind_cnt + vertices.size() - 1) % vertices.size()]);
    second_segment =
        Point(vertices[(ind_cnt + 1) % vertices.size()] - vertices[ind_cnt]);
    auto res = VectorMul(first_segment, second_segment);
    if (res != 0) {
      orientation = res > 0;
      break;
    }
    if (ind_cnt >= vertices.size()) {
      return true;
    }
    ++ind_cnt;
  }

  for (size_t index = ind_cnt + 1; index < vertices.size() + ind_cnt + 1;
       ++index) {
    size_t next = (index + 1) % vertices.size();
    first_segment = second_segment;
    second_segment = vertices[next] - vertices[index % vertices.size()];

    auto res = VectorMul(first_segment, second_segment);
    if (res != 0 && orientation != (res > 0)) {
      return false;
    }
  }
  return true;
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);
  static const std::streamsize kISendCodestyleToTheThreeFunnyLetters = 20;
  std::cout.precision(kISendCodestyleToTheThreeFunnyLetters);

  size_t cnt;
  std::cin >> cnt;
  std::vector<Point> ptt(cnt);
  std::cin >> ptt;

  std::cout << (IsConvex(ptt) ? "YES" : "NO") << '\n';

  std::cout.flush();
  return 0;
}
