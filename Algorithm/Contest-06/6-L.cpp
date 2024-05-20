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

struct Algo {
  long long k, v;
  Algo() : k(0), v(0) {}
  Algo(size_t kn, size_t vn) : k(kn), v(vn) {}
};

int GetOptimalSequence(const std::vector<std::vector<Algo>>& arr, size_t res) {
  std::vector<std::vector<long long>> dp(arr.size() + 1,
                                         std::vector<long long>(res + 1, -1));

  dp[0][res] = 0;
  for (size_t last = 1; last <= arr.size(); ++last) {
    for (size_t index = 0; index < arr[last - 1].size(); ++index) {
      for (size_t rest = 0; rest <= res; ++rest) {
        if (dp[last - 1][rest] != -1) {
          dp[last][rest] = std::max(dp[last][rest], dp[last - 1][rest]);
        }

        size_t prev_rest = rest + arr[last - 1][index].k;
        if (prev_rest <= res && dp[last - 1][prev_rest] != -1 &&
            dp[last][rest] < dp[last - 1][prev_rest] + arr[last - 1][index].v) {
          dp[last][rest] = dp[last - 1][prev_rest] + arr[last - 1][index].v;
        }
      }
    }
  }
  
  long long optimal_rest = 0;
  for (const auto& p : dp[arr.size()]) {
    optimal_rest = std::max(optimal_rest, p);
  }
  return optimal_rest;
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t n, g, k;
  std::cin >> n >> g >> k;
  std::vector<std::vector<Algo>> arr(g);
  for (size_t i = 0; i < n; ++i) {
    long long k, v, g;
    std::cin >> k >> v >> g;
    arr[g - 1].push_back(Algo(k, v));
  }

  std::cout << GetOptimalSequence(arr, k);

  return 0;
}
