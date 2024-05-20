/*

*/

#include <algorithm>
#include <cmath>
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

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t n = 0;
  std::cin >> n;

  std::vector<std::vector<long long>> roads(n, std::vector<long long>(n));
  for (auto& str : roads) {
    std::cin >> str;
  }

  size_t max_mask = (1 << n);
  std::vector<std::vector<long long>> dp(max_mask,
                                         std::vector<long long>(n, 1e18));
  std::vector<std::vector<std::pair<size_t, size_t>>> pr(
      max_mask, std::vector<std::pair<size_t, size_t>>(n, {0, n}));
  for (size_t index = 0; index < n; ++index) {
    dp[1 << index][index] = 0;
  }

  for (size_t mask = 2; mask < max_mask; mask++) {
    for (size_t index = 0; index < n; index++) {
      if (((mask >> index) & 1) == 1) {
        size_t mask_prev = mask ^ (1 << index);

        for (size_t other = 0; other < n; other++) {
          if (other != index && ((mask >> other) & 1) == 1) {
            long long new_val = dp[mask_prev][other] + roads[index][other];
            if (dp[mask][index] > new_val) {
              dp[mask][index] = new_val;
              pr[mask][index] = {mask_prev, other};
            }
          }
        }
      }
    }
  }

  std::pair<size_t, size_t> ans = {max_mask - 1, 0};
  for (size_t index = 0; index < n; index++) {
    if (dp[ans.first][ans.second] > dp[ans.first][index]) {
      ans.second = index;
    }
  }

  std::cout << dp[ans.first][ans.second] << '\n';

  std::vector<size_t> way;
  while (ans.second != n) {
    way.push_back(ans.second + 1);
    ans = pr[ans.first][ans.second];
  }

  std::cout << way;

  return 0;
}
