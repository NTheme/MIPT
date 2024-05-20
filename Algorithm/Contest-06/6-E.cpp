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

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t n;
  std::cin >> n;

  std::vector<std::vector<int>> dp(n + 1, std::vector<int>(n + 1));
  std::vector<std::vector<int>> pr(n + 1, std::vector<int>(n + 1));

  for (size_t i = 1; i <= n; ++i) {
    dp[i][i] = 1;
    for (size_t j = 1; j < i; ++j) {
      dp[i][j] = pr[i - j][j / 2];
    }
    pr[i][0] = dp[i][0];
    for (size_t j = 1; j <= n; ++j) {
      pr[i][j] = pr[i][j - 1] + dp[i][j];
    }
  }

  size_t ans = 0;
  for (const auto& p : dp[n]) {
    ans += p;
  }

  std::cout << ans << '\n';

  return 0;
}
