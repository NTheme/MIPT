/*

*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <algorithm>
#include <iostream>
#include <vector>

using std::pair;
using std::vector;

template <typename TypeFirst, typename TypeSecond>
std::istream& operator>>(std::istream& inp, pair<TypeFirst, TypeSecond>& pair) {
  inp >> pair.first >> pair.second;
  return inp;
}
template <typename TypeFirst, typename TypeSecond>
std::ostream& operator<<(std::ostream& out,
                         const pair<TypeFirst, TypeSecond>& pair) {
  out << pair.first << ' ' << pair.second;
  return out;
}

template <typename Type>
std::istream& operator>>(std::istream& inp, vector<Type>& array) {
  for (auto& elem : array) {
    inp >> elem;
  }
  return inp;
}
template <typename Type>
std::ostream& operator<<(std::ostream& out, const vector<Type>& array) {
  for (const auto& elem : array) {
    out << elem << '\n';
  }
  return out;
}

vector<vector<size_t>> graph;
vector<int> set_pairs;
vector<bool> used;

bool Kunn(size_t ver) {
  if (used[ver]) {
    return false;
  }
  used[ver] = true;
  for (const auto& to : graph[ver]) {
    if (set_pairs[to] == -1 || Kunn(set_pairs[to])) {
      set_pairs[to] = ver;
      return true;
    }
  }
  return false;
}

void DFS(size_t ver) {
  used[ver] = true;

  for (const auto& to : graph[ver]) {
    if (!used[to]) {
      DFS(to);
    }
  }
}

void Task() {
  size_t num_ver;
  size_t num_edge;
  std::cin >> num_ver >> num_edge;

  graph.resize(num_edge);
  used.assign(num_edge, false);
  set_pairs.assign(num_edge, -1);

  for (size_t index = 0; index < num_edge; ++index) {
    size_t from, to;
    std::cin >> from >> to;
    graph[from - 1].push_back(to - 1);
  }

  for (size_t ver = 0; ver < num_ver; ++ver) {
    used.assign(num_ver, false);
    Kunn(ver);
  }

  graph.assign(num_ver, {});
  used.assign(num_ver, false);
  for (size_t ver = 0; ver < num_ver; ++ver) {
    if (set_pairs[ver] != -1) {
      graph[set_pairs[ver]].push_back(ver);
      graph[ver].push_back(set_pairs[ver]);
    }
  }

  size_t ans = 0;
  for (size_t ind = 0; ind < num_ver; ++ind) {
    if (!used[ind] && graph[ind].size() < 2) {
      DFS(ind);
      ++ans;
    }
  }
  std::cout << ans << '\n';
}

signed main() {
  std::cin.tie(nullptr)->sync_with_stdio(false);
  std::cout.precision(20);

  Task();

  std::cout.flush();
  return 0;
}
