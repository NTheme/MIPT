#include <algorithm>
#include <iostream>
#include <vector>

using std::cin;
using std::cout;
using std::vector;

vector<vector<int>> g;
vector<bool> u;

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

vector<int> Dfs(int v) {
  u[v] = true;

  vector<int> ans;
  ans.push_back(v + 1);

  for (size_t i = 0; i < g[v].size(); i++) {
    if (!u[g[v][i]]) {
      auto pb = Dfs(g[v][i]);
      copy(pb.begin(), pb.end(), back_inserter(ans));
    }
  }

  return ans;
}

signed main() {
  cin.tie(0)->sync_with_stdio(false);
  cout.precision(20);

  int n = 0, m = 0;
  cin >> n >> m;

  g.resize(n);
  u.assign(n, false);

  for (int i = 0; i < m; i++) {
    int b1, b2;
    cin >> b1 >> b2;

    g[b1 - 1].push_back(b2 - 1);
    g[b2 - 1].push_back(b1 - 1);
  }

  vector<vector<int>> ans;
  for (int i = 0; i < n; i++) {
    if (!u[i]) {
      ans.push_back(Dfs(i));
    }
  }

  cout << ans.size() << '\n';
  for (size_t i = 0; i < ans.size(); i++) {
    cout << ans[i].size() << '\n' << ans[i] << '\n';
  }

  return 0;
}