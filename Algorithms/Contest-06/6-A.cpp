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

void Solve(int n, int m,
           std::vector<std::vector<std::pair<long long, std::string>>> ppl) {
  std::vector<std::vector<long long>> dp(n + 1,
                                         std::vector<long long>(m + 1, 1e9));
  std::vector<std::vector<std::pair<int, int>>> pr(
      n + 1, std::vector<std::pair<int, int>>(m + 1, {0, 0}));
  dp[0][0] = 0;
  for (int i = 1; i <= n; ++i) {
    for (int j = 0; j <= m; ++j) {
      long long max_b = 0;
      for (int b = 1; b <= 4 && b <= i; ++b) {
        max_b = std::max(max_b, ppl[0][i - b].first);
        long long max_g = 0;
        for (int g = 0; g <= 4 - b && g <= j; ++g) {
          if (g != 0) {
            max_g = std::max(max_g, ppl[1][j - g].first);
          }
          if (dp[i][j] > dp[i - b][j - g] + std::max(max_b, max_g)) {
            dp[i][j] =
                std::min(dp[i][j], dp[i - b][j - g] + std::max(max_b, max_g));
            pr[i][j] = {i - b, j - g};
          }
        }
      }
    }
  }
  std::cout << dp[n][m] << '\n';
  std::vector<std::vector<std::pair<int, int>>> tx;
  std::pair<int, int> best = {n, m};
  while (best.first != 0 || best.second != 0) {
    tx.push_back({});
    for (int i = pr[best.first][best.second].first; i < best.first; ++i) {
      tx.back().push_back({0, i});
    }
    for (int j = pr[best.first][best.second].second; j < best.second; ++j) {
      tx.back().push_back({1, j});
    }
    best = pr[best.first][best.second];
  }
  std::reverse(tx.begin(), tx.end());
  std::cout << tx.size() << '\n';
  for (int i = 0; i < (int)tx.size(); ++i) {
    std::cout << "Taxi " << i + 1 << ": ";
    for (int j = 0; j < (int)tx[i].size(); ++j) {
      std::cout << ppl[tx[i][j].first][tx[i][j].second].second;
      if (j < (int)tx[i].size() - 2) {
        std::cout << ", ";
      } else if (j == (int)tx[i].size() - 2) {
        std::cout << " and ";
      } else {
        std::cout << ".\n";
      }
    }
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  int n, m;
  std::cin >> n;
  std::vector<std::vector<std::pair<long long, std::string>>> ppl(2);
  ppl[0].resize(n);
  for (auto& p : ppl[0]) {
    std::cin >> p.second >> p.first;
  }
  std::cin >> m;
  ppl[1].resize(m);
  for (auto& p : ppl[1]) {
    std::cin >> p.second >> p.first;
  }
  std::sort(ppl[0].begin(), ppl[0].end());
  std::sort(ppl[1].begin(), ppl[1].end());
  Solve(n, m, ppl);
  return 0;
}
