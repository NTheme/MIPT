/*

*/

#include <algorithm>
#include <iostream>
#include <vector>

using std::pair;
using std::vector;

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

vector<int> val;
vector<vector<pair<size_t, int>>> g;
vector<bool> u;

int max_1 = -1e9, max_2 = -1e9;

void Dfs(size_t v, bool more) {
  u[v] = true;
  if (more) {
    max_1 = std::max(max_1, val[v]);
  } else {
    max_2 = std::max(max_2, val[v]);
  }

  for (const auto& p : g[v]) {
    if (!u[p.first]) {
      val[p.first] = p.second - val[v];
      Dfs(p.first, !more);
    }
  }
}

int main() {
  std::cin.tie(nullptr)->sync_with_stdio(false);
  std::cout.precision(20);

  size_t n, m;
  std::cin >> n >> m;

  g.resize(n);
  val.assign(n, 0);

  for (size_t i = 0; i < m; ++i) {
    size_t u, v, c;
    std::cin >> u >> v >> c;
    g[u - 1].push_back({v - 1, c});
    g[v - 1].push_back({u - 1, c});
  }

  u.assign(n, false);
  val[0] = 1;
  max_1 = -1e9, max_2 = -1e9;
  Dfs(0, true);

  if (max_1 < (int)n) {
    u.assign(n, false);
    val[0] = n - max_1 + 1;
    max_1 = -1e9, max_2 = -1e9;
    Dfs(0, true);
    auto vv = val;
    std::sort(vv.begin(), vv.end());
    for (size_t i = 1; i <= n; ++i) {
      if (vv[i - 1] != (int)i) {
        u.assign(n, false);
        val[0] = 1;
        max_1 = -1e9, max_2 = -1e9;
        Dfs(0, true);

        u.assign(n, false);
        val[0] = 1 + max_2 - n;
        max_1 = -1e9, max_2 = -1e9;
        Dfs(0, true);
        break;
      }
    }
  }

  std::cout << val;

  return 0;
}
