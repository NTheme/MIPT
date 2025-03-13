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

size_t LCIS(const std::vector<int>& first_arr,
            const std::vector<int>& second_arr) {
  std::vector<std::vector<size_t>> dp(
      first_arr.size() + 1, std::vector<size_t>(second_arr.size() + 1));

  for (size_t first = 1; first <= first_arr.size(); ++first) {
    std::pair<size_t, size_t> best;
    for (size_t second = 1; second <= second_arr.size(); ++second) {
      dp[first][second] = dp[first - 1][second];
      if (first_arr[first - 1] == second_arr[second - 1] &&
          dp[first - 1][second] < best.second + 1) {
        dp[first][second] = best.second + 1;
      }
      if (first_arr[first - 1] > second_arr[second - 1] &&
          dp[first - 1][second] > best.second) {
        best = {second, dp[first - 1][second]};
      }
    }
  }

  size_t res = 1;
  for (size_t index = 1; index <= second_arr.size(); ++index) {
    if (dp[first_arr.size()][res] < dp[first_arr.size()][index]) {
      res = index;
    }
  }

  return dp[first_arr.size()][res];
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t n, m;
  std::cin >> n >> m;
  std::vector<int> first_arr(n), second_arr(m);
  std::cin >> first_arr >> second_arr;

  std::cout << LCIS(first_arr, second_arr);

  return 0;
}
