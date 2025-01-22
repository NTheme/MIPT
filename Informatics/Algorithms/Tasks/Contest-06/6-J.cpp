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
  std::vector<long long> array(n);
  std::cin >> array;

  const int kMod = 1e9 + 7;
  std::vector<long long> last(1000000, -1);
  std::vector<long long> dp(n + 1);

  dp[0] = 1;
  for (size_t index = 1; index <= n; index++) {
    dp[index] = (2 * dp[index - 1]) % kMod;
    if (last[array[index - 1]] != -1) {
      dp[index] = (dp[index] - dp[last[array[index - 1]]] + kMod) % kMod;
    }

    last[array[index - 1]] = index - 1;
  }

  std::cout << (dp[n] + kMod - 1) % kMod << '\n';

  return 0;
}
