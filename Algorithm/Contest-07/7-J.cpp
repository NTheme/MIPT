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
    out << elem;
  }
  return out;
}

vector<vector<pair<size_t, size_t>>> g;
vector<bool> u;
std::string s;
vector<bool> e;

void Dfs(size_t v, bool& make) {
  u[v] = true;
  if (s[v] == '1') {
    make = !make;
  }

  for (const auto& p : g[v]) {
    if (!u[p.first]) {
      e[p.second] = e[p.second] ^ make;
      Dfs(p.first, make);
      e[p.second] = e[p.second] ^ make;
    }
  }
}

int main() {
  std::cin.tie(nullptr)->sync_with_stdio(false);
  std::cout.precision(20);

  size_t t;
  std::cin >> t;
  while (t-- > 0) {
    size_t n, m;
    std::cin >> n >> m;

    g.assign(n, {});
    u.assign(n, false);
    s.clear();
    e.assign(m, false);

    for (size_t i = 0; i < m; ++i) {
      size_t u, v;
      std::cin >> u >> v;
      g[u - 1].push_back({v - 1, i});
      g[v - 1].push_back({u - 1, i});
    }

    std::cin >> s;

    bool cond = true;
    for (size_t i = 0; i < n; ++i) {
      if (!u[i]) {
        bool make = false;
        Dfs(i, make);
        if (make) {
          cond = false;
          break;
        }
      }
    }

    if (cond) {
      std::cout << e << '\n';
    } else {
      std::cout << "-1\n";
    }
  }

  return 0;
}
