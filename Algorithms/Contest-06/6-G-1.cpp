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

std::vector<int> BandSequence(const std::vector<int>& array) {
  std::vector<std::vector<size_t>> dp(array.size(), std::vector<size_t>(2, 1));
  std::vector<std::vector<std::pair<int, int>>> pr(
      array.size(), std::vector<std::pair<int, int>>(2, {-1, -1}));

  dp[0][0] = dp[0][1] = 1;

  for (size_t index = 1; index < array.size(); ++index) {
    for (size_t value = 0; value < index; ++value) {
      if (array[index] < array[value] && dp[index][1] < dp[value][0] + 1) {
        dp[index][1] = dp[value][0] + 1;
        pr[index][1] = {value, 0};
      }
      if (array[index] > array[value] && dp[index][0] < dp[value][1] + 1) {
        dp[index][0] = dp[value][1] + 1;
        pr[index][0] = {value, 1};
      }
    }
  }

  std::pair<int, int> best = {0, 0};
  for (size_t index = 0; index < array.size(); ++index) {
    if (dp[best.first][best.second] < dp[index][0]) {
      best = {index, 0};
    }
    if (dp[best.first][best.second] < dp[index][1]) {
      best = {index, 1};
    }
  }
  std::vector<int> ret;
  while (best.first != -1) {
    ret.push_back(array[best.first]);
    best = pr[best.first][best.second];
  }

  std::reverse(ret.begin(), ret.end());

  return ret;
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t n;
  std::cin >> n;
  std::vector<int> array(n);
  std::cin >> array;

  auto ss = BandSequence(array);
  std::cout << ss.size() << '\n' << ss;

  return 0;
}
