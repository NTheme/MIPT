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

  int n;
  std::cin >> n;
  std::vector<std::vector<char>> arr(n, std::vector<char>(n));
  for (auto& p : arr) {
    std::cin >> p;
  }

  std::vector<bool> dp(1 << n, false);
  dp[0] = true;

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < (1 << i); ++j) {
      if (dp[j]) {
        for (int k = 0; k < i; ++k) {
          if (arr[i][k] == 'Y' && (j & (1 << k)) == 0) {
            dp[j + (1 << k) + (1 << i)] = true;
          }
        }
      }
    }
  }

  int max_ps = 0;
  for (int i = 0; i < (1 << n); ++i) {
    if (dp[i]) {
      int pws = 0;
      int mask = i;
      while (mask > 0) {
        pws += mask % 2;
        mask >>= 1;
      }
      max_ps = std::max(pws, max_ps);
    }
  }
  std::cout << max_ps << '\n';

  return 0;
}
