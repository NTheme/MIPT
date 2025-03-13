#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using std::cin;
using std::cout;
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

vector<vector<int>> g;
vector<int> u;
vector<bool> us;
vector<int> ans;

int n = 0, m = 0, cs = -1;

bool Dfs(int v) {
  u[v] = 1;
  for (size_t i = 0; i < g[v].size(); i++) {
    int to = g[v][i];
    if (u[to] == 0) {
      if (Dfs(to)) {
        return true;
      }
    } else if (u[to] == 1) {
      cs = to;
      return true;
    }
  }
  u[v] = 2;
  return false;
}

void Dfs2(int v) {
  us[v] = true;
  for (size_t i = 0; i < g[v].size(); i++) {
    int to = g[v][i];
    if (!us[to]) {
      Dfs2(to);
    }
  }
  ans.push_back(v + 1);
}

signed main() {
  cin.tie(0)->sync_with_stdio(false);
  cout.precision(20);

  cin >> n >> m;

  u.assign(n, 0);
  us.assign(n, false);
  g.resize(n);

  for (int i = 0; i < m; i++) {
    int b1 = 0, b2 = 0;
    cin >> b1 >> b2;
    g[b1 - 1].push_back(b2 - 1);
  }

  for (int i = 0; i < n; i++) {
    if (Dfs(i)) {
      break;
    }
  }

  if (cs == -1) {
    for (int i = 0; i < n; i++) {
      if (!us[i]) {
        Dfs2(i);
      }
    }
    std::reverse(ans.begin(), ans.end());
    cout << ans << '\n';
  } else {
    cout << -1 << '\n';
  }

  return 0;
}
