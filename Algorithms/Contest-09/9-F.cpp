/*

*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <algorithm>
#include <vector>
#include <iostream>
#include <memory>

using std::vector;
using std::pair;
using std::shared_ptr;
using std::string;

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
    out << elem << ' ';
  }
  return out;
}

vector<vector<int>> graph, ver_up;
vector<int> dp, time_in, time_out, parent, alive;
vector<bool> deleted;
int timer = 0, max_height = 19;

struct Query {
  char type = '\0';
  int ver1 = 0, ver2 = 0;
};

std::istream& operator>>(std::istream& inp, Query& query) {
  inp >> query.type >> query.ver1;
  --query.ver1;
  if (query.type == '?') {
    inp >> query.ver2;
    --query.ver2;
  }
  return inp;
}

bool IsAncestor(int ver1, int ver2) {
  return time_in[ver1] <= time_in[ver2] && time_out[ver1] >= time_out[ver2];
}

void DFS(int ver1, int parent = -1) {
  for (int ind = 1; ind < max_height; ++ind) {
    ver_up[ver1][ind] = ver_up[ver_up[ver1][ind - 1]][ind - 1];
  }
  time_in[ver1] = timer++;
  for (int ver2 : graph[ver1]) {
    if (ver2 != parent) {
      dp[ver2] = dp[ver1] + 1;
      ver_up[ver2][0] = ver1;
      DFS(ver2, ver1);
    }
  }
  time_out[ver1] = timer++;
}

int LCA(int ver1, int ver2) {
  if (IsAncestor(ver1, ver2)) {
    return ver1;
  }
  if (IsAncestor(ver2, ver1)) {
    return ver2;
  }
  for (int ind = max_height - 1; ind >= 0; --ind) {
    if (!IsAncestor(ver_up[ver1][ind], ver2)) {
      ver1 = ver_up[ver1][ind];
    }
  }
  return ver_up[ver1][0];
}

int FindNotDel(int ver1) {
  if (!deleted[alive[ver1]]) {
    return alive[ver1];
  }
  alive[ver1] = FindNotDel(parent[alive[ver1]]);
  return alive[ver1];
}

void Task() {
  int num_que;
  std::cin >> num_que;

  graph.resize(1);
  parent.resize(1, -1);
  vector<Query> queries(num_que);

  for (auto& query : queries) {
    std::cin >> query;
    if (query.type == '+') {
      graph.push_back({query.ver1});
      parent.push_back(query.ver1);
      graph[query.ver1].push_back(graph.size() - 1);
    }
  }

  ver_up.assign(graph.size(), vector<int>(max_height, 0));
  dp.resize(graph.size(), 0);
  time_in.resize(graph.size(), 0);
  time_out.resize(graph.size(), 0);
  deleted.resize(graph.size());
  for (int ind = 0; ind < (int)graph.size(); ++ind) {
    alive.push_back(ind);
  }

  DFS(0);
  for (const auto& query : queries) {
    if (query.type == '-') {
      deleted[query.ver1] = true;
    } else if (query.type == '?') {
      auto common = LCA(query.ver1, query.ver2);
      std::cout << FindNotDel(common) + 1 << '\n';
    }
  }
}

signed main() {
  std::cin.tie(nullptr)->sync_with_stdio(false);
  std::cout.precision(20);

  Task();

  std::cout.flush();
  return 0;
}
