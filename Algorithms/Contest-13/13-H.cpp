/******************************************
 *  Author : NThemeDEV
 *  Created : Thu Nov 23 2023
 *  File : 13-H.cpp
 ******************************************/

/*
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")
*/

#include <algorithm>
#include <iomanip>
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
    out << elem << '\n';
  }
  return out;
}

using Val = long long;
using Point = std::pair<Val, Val>;

bool operator<(const Point& lhs, const Point& rhs) {
  return lhs.first < rhs.first ||
         (lhs.first == rhs.first && lhs.second < rhs.second);
}

Val OrientedSq(const Point& aa, const Point& bb, const Point& cc) {
  return aa.first * (bb.second - cc.second) +
         bb.first * (cc.second - aa.second) +
         cc.first * (aa.second - bb.second);
}

std::vector<Point> ConvexHull(std::vector<Point> pt) {
  std::sort(pt.begin(), pt.end());
  Point pt_d = pt[0];
  Point pt_u = pt.back();

  std::vector<Point> up = {pt_d};
  std::vector<Point> dn = {pt_d};

  for (size_t ind = 1; ind < pt.size(); ++ind) {
    if (ind == pt.size() - 1 || OrientedSq(pt_d, pt[ind], pt_u) < 0) {
      while (up.size() >= 2 &&
             OrientedSq(up[up.size() - 2], up[up.size() - 1], pt[ind]) >= 0) {
        up.pop_back();
      }
      up.push_back(pt[ind]);
    }
    if (ind == pt.size() - 1 || OrientedSq(pt_d, pt[ind], pt_u) > 0) {
      while (dn.size() >= 2 &&
             OrientedSq(dn[dn.size() - 2], dn[dn.size() - 1], pt[ind]) <= 0) {
        dn.pop_back();
      }
      dn.push_back(pt[ind]);
    }
  }

  pt.clear();
  for (size_t ind = 0; ind < up.size(); ++ind) {
    pt.push_back(up[ind]);
  }
  for (size_t ind = dn.size() - 2; ind > 0; --ind) {
    pt.push_back(dn[ind]);
  }

  return pt;
}

uint64_t ConvexSq(const std::vector<Point>& pt) {
  uint64_t sq = 0;
  for (size_t ind = 1; ind < pt.size() - 1; ++ind) {
    sq += std::abs(OrientedSq(pt[0], pt[ind], pt[ind + 1]));
  }

  return sq;
}

std::string ToOutput(uint64_t val) {
  static const uint64_t kBase = 10;

  std::string str;
  str += '0' + 5 * (val % 2);
  str += '.';

  val /= 2;
  while (val > 0) {
    str += '0' + val % kBase;
    val /= kBase;
  }
  std::reverse(str.begin(), str.end());
  return str;
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);

  size_t num;
  std::cin >> num;
  std::vector<Point> pt(num);
  std::cin >> pt;

  auto convex_hull = ConvexHull(pt);
  std::cout << convex_hull.size() << '\n'
            << convex_hull << ToOutput(ConvexSq(convex_hull)) << '\n';

  std::cout.flush();
  return 0;
}
