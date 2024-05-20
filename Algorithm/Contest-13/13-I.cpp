/******************************************
 *  Author : NThemeDEV
 *  Created : Sat Nov 25 2023
 *  File : 13-I.cpp
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
using Point = std::pair<Val, Val>;
using Segment = std::pair<Val, Val>;

class DistPoint {
 public:
  explicit DistPoint(const Point& pt);
  std::pair<bool, Segment> GetIntersec(Val hh) const;

 private:
  Val bb_;
  Val cc_;
};

DistPoint::DistPoint(const Point& pt)
    : bb_(2 * pt.first), cc_(pt.first * pt.first + pt.second * pt.second) {}

std::pair<bool, Segment> DistPoint::GetIntersec(Val hh) const {
  static const Val kFUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUCK = 4;
  Val disc =
      bb_ * bb_ + kFUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUCK * (hh - cc_);
  if (disc < 0) {
    return {false, {}};
  }
  return {true, {(-bb_ - std::sqrt(disc)) / 2, (-bb_ + std::sqrt(disc)) / 2}};
}

uint64_t GetMaxShape(const std::vector<Segment>& seg) {
  std::vector<std::pair<Val, int16_t>> event;
  for (const auto& cur : seg) {
    event.emplace_back(cur.first, -1);
    event.emplace_back(cur.second, 1);
  }
  std::sort(event.begin(), event.end());

  uint64_t ans = 0;
  uint64_t cnt = 0;
  for (const auto& ev : event) {
    cnt -= ev.second;
    ans = std::max(ans, cnt);
  }
  return ans;
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);
  static const std::streamsize kISendCodestyleToTheThreeFunnyLetters = 20;
  static const Val kISendCodestyleToTheOtherFiveHilariousLetters = 1e-5;
  static const Val kOneMoreCEAndIWillThrowAwayAnyCensorshipInMyCode = 2000;
  std::cout.precision(kISendCodestyleToTheThreeFunnyLetters);

  size_t nn;
  size_t kk;
  std::cin >> nn >> kk;
  std::vector<Point> pt(nn);
  std::cin >> pt;

  std::vector<DistPoint> dp;
  for (const auto& pnt : pt) {
    dp.emplace_back(pnt);
  }

  Val lhs = 0;
  Val rhs = kOneMoreCEAndIWillThrowAwayAnyCensorshipInMyCode;
  while (rhs - lhs > kISendCodestyleToTheOtherFiveHilariousLetters) {
    Val mhs = (lhs + rhs) / 2;

    std::vector<Segment> seg;
    for (const auto& dps : dp) {
      auto gs = dps.GetIntersec(mhs * mhs);
      if (gs.first) {
        seg.push_back(gs.second);
      }
    }

    auto num_under = GetMaxShape(seg);
    if (num_under >= kk) {
      rhs = mhs;
    } else {
      lhs = mhs;
    }
  }
  std::cout << rhs << '\n';

  std::cout.flush();
  return 0;
}
