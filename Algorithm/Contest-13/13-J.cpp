/******************************************
 *  Author : NThemeDEV
 *  Created : Sat Nov 25 2023
 *  File : 13-J.cpp
 ******************************************/

/*
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")
*/

#include <algorithm>
#include <array>
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

static const size_t kNumPolygons = 3;

using Val = long double;
using Vector = std::pair<Val, Val>;
using Point = std::pair<Val, Val>;

bool Comp1(const Point& lhs, const Point& rhs) {
  return lhs.second < rhs.second ||
         (lhs.second == rhs.second && lhs.first < rhs.first);
}

bool Comp2(const Point& lhs, const Point& rhs) {
  return lhs.first < rhs.first ||
         (lhs.first == rhs.first && lhs.second < rhs.second);
}

bool Comp3(const Vector& lhs, const Vector& rhs) {
  if (lhs.second == 0 && rhs.second == 0) {
    return lhs.first < rhs.first;
  }
  return lhs.first * rhs.second < lhs.second * rhs.first;
}

Point& operator*=(Point& pt, size_t num) {
  pt.first *= num;
  pt.second *= num;
  return pt;
}

Vector operator+(Vector lhs, const Vector& rhs) {
  lhs.first += rhs.first;
  lhs.second += rhs.second;
  return lhs;
}

Vector operator-(Vector lhs, const Vector& rhs) {
  lhs.first -= rhs.first;
  lhs.second -= rhs.second;
  return lhs;
}

Val CrossMultiply(const Vector& lhs, const Vector& rhs) {
  return lhs.first * rhs.second - lhs.second * rhs.first;
}

Val ScalarMultiply(const Vector& lhs, const Vector& rhs) {
  return lhs.first * rhs.first + lhs.second * rhs.second;
}

Val OrientedSq(const Point& aa, const Point& bb, const Point& cc) {
  return aa.first * (bb.second - cc.second) +
         bb.first * (cc.second - aa.second) +
         cc.first * (aa.second - bb.second);
}

void ReorderPolygon(std::vector<Point>& plg,
                    bool comp(const Point& lhs, const Point& rhs)) {
  size_t pos = 0;
  for (size_t ind = 1; ind < plg.size(); ++ind) {
    if (comp(plg[ind], plg[pos])) {
      pos = ind;
    }
  }
  std::rotate(plg.begin(), plg.begin() + pos, plg.end());
}

std::vector<Point> GetMinkowskiSum(std::vector<Point> firrst,
                                   std::vector<Point> second) {
  std::vector<Point> result;

  firrst.push_back(firrst[0]);
  firrst.push_back(firrst[1]);
  second.push_back(second[0]);
  second.push_back(second[1]);

  size_t lhs = 0;
  size_t rhs = 0;
  while (lhs < firrst.size() - 2 || rhs < second.size() - 2) {
    result.push_back(firrst[lhs] + second[rhs]);
    auto cross = CrossMultiply(firrst[lhs + 1] - firrst[lhs],
                               second[rhs + 1] - second[rhs]);
    if (cross >= 0 && lhs < firrst.size() - 2) {
      ++lhs;
    }
    if (cross <= 0 && rhs < second.size() - 2) {
      ++rhs;
    }
  }
  return result;
}

std::vector<Point> GetMiddlePolygon(
    std::array<std::vector<Point>, kNumPolygons> plg) {
  for (auto& polygon : plg) {
    ReorderPolygon(polygon, Comp1);
  }

  auto middle = GetMinkowskiSum(plg[0], GetMinkowskiSum(plg[1], plg[2]));
  ReorderPolygon(middle, Comp2);
  return middle;
}

std::vector<Vector> GetAngleArray(const std::vector<Point>& middle) {
  std::vector<Vector> angles(middle.size() - 1);
  for (size_t ind = 1; ind < middle.size(); ++ind) {
    angles[ind - 1].first = middle[ind].second - middle[0].second;
    angles[ind - 1].second = middle[ind].first - middle[0].first;
    if (angles[ind - 1].first == 0) {
      angles[ind - 1].second = angles[ind - 1].second < 0 ? -1 : 1;
    }
  }
  return angles;
}

bool ContainsPoint(const std::vector<Point>& middle,
                   const std::vector<Vector>& angles, const Point& cur) {
  bool inside = false;
  if (cur.first >= middle[0].first) {
    if (cur.first == middle[0].first && cur.second == middle[0].second) {
      inside = true;
    } else {
      Vector to_zero = {cur.second - middle[0].second,
                        cur.first - middle[0].first};
      if (to_zero.first == 0) {
        to_zero.second = to_zero.second < 0 ? -1 : 1;
      }
      auto triangular =
          std::upper_bound(angles.begin(), angles.end(), to_zero, Comp3);
      if (triangular == angles.end() && to_zero.first == angles.back().first &&
          to_zero.second == angles.back().second) {
        triangular = angles.end() - 1;
      }
      if (triangular != angles.end() && triangular != angles.begin()) {
        size_t p1 = triangular - angles.begin();
        if (OrientedSq(middle[p1 + 1], middle[p1], cur) <= 0) {
          inside = true;
        }
      }
    }
  }
  return inside;
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);

  std::array<std::vector<Point>, kNumPolygons> plg;
  for (auto& polygon : plg) {
    size_t nn;
    std::cin >> nn;
    polygon.resize(nn);
    std::cin >> polygon;
  }

  auto middle = GetMiddlePolygon(plg);
  auto angles = GetAngleArray(middle);

  size_t mm;
  std::cin >> mm;
  while (mm-- > 0) {
    Point cur;
    std::cin >> cur;
    cur *= 3;

    std::cout << (ContainsPoint(middle, angles, cur) ? "YES\n" : "NO\n");
  }

  std::cout.flush();
  return 0;
}
