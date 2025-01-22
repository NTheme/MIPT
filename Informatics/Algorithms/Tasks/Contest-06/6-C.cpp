/*

*/

#include <algorithm>
#include <iostream>
#include <vector>

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

std::vector<long long> BandSequence(const std::vector<long long>& coord,
                                    size_t m) {
  std::vector<std::vector<long long>> dp(
      m, std::vector<long long>(coord.size(), 1e9));
  std::vector<std::vector<long long>> previ(
      m, std::vector<long long>(coord.size(), -1));

  dp[0][0] = 0;
  for (size_t index = 1; index < coord.size(); ++index) {
    dp[0][index] = dp[0][index - 1] + index * (coord[index] - coord[index - 1]);
  }

  for (size_t num = 1; num < m; ++num) {
    for (size_t last = num; last < coord.size(); ++last) {
      for (size_t prev = num - 1; prev < last; ++prev) {
        long long new_val = dp[num - 1][prev];
        for (size_t index = prev + 1; index < last; ++index) {
          new_val += std::min(coord[index] - coord[prev],
                              std::abs(coord[index] - coord[last]));
        }
        if (new_val < dp[num][last]) {
          dp[num][last] = new_val;
          previ[num][last] = prev;
        }
      }
    }
  }

  for (size_t index = 0; index < coord.size(); ++index) {
    for (size_t i = index; i < coord.size(); ++i) {
      dp[m - 1][index] += coord[i] - coord[index];
    }
  }
  long long min_index = m - 1;
  for (size_t index = m - 1; index < coord.size(); ++index) {
    if (dp[m - 1][min_index] > dp[m - 1][index]) {
      min_index = index;
    }
  }

  long long ss = min_index, mm = m;
  std::vector<long long> ret;
  while (min_index != -1) {
    ret.push_back(coord[min_index]);
    min_index = previ[m - 1][min_index];
    --m;
  }
  ret.push_back(dp[mm - 1][ss]);

  std::reverse(ret.begin(), ret.end());

  return ret;
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t n, m;
  std::cin >> n >> m;
  std::vector<long long> coord(n);
  std::cin >> coord;

  auto ss = BandSequence(coord, m);
  std::cout << ss;

  return 0;
}
