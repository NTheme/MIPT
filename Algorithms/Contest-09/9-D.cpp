/*

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

*/

#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

using std::pair;
using std::string;
using std::vector;

template <typename TypeFirst, typename TypeSecond>
std::istream& operator>>(std::istream& inp, pair<TypeFirst, TypeSecond>& pair) {
  inp >> pair.first >> pair.second;
  --pair.first;
  --pair.second;
  return inp;
}
template <typename TypeFirst, typename TypeSecond>
std::ostream& operator<<(std::ostream& out, const pair<TypeFirst, TypeSecond>& pair) {
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

vector<vector<size_t>> graph;
vector<bool> used;
vector<size_t> dist;
vector<size_t> dsu;
vector<size_t> parent;
vector<vector<pair<size_t, size_t>>> asks;
vector<size_t> ans;
vector<bool> used2;

void DFS(size_t ver) {
  used2[ver] = true;
  for (const auto& to : graph[ver]) {
    if (!used2[to]) {
      dist[to] = dist[ver] + 1;
      DFS(to);
    }
  }
}

size_t Dsu_get(size_t ver) { return ver == dsu[ver] ? ver : dsu[ver] = Dsu_get(dsu[ver]); }

void Dsu_unite(size_t ver1, size_t ver2, size_t new_ancestor) {
  ver1 = Dsu_get(ver1), ver2 = Dsu_get(ver2);
  if ((rand() & 1) == 1) {
    std::swap(ver1, ver2);
  }
  dsu[ver1] = ver2;
  parent[ver2] = new_ancestor;
}

void Dfs(size_t ver) {
  dsu[ver] = ver;
  parent[ver] = ver;
  used[ver] = true;

  for (const auto& to : graph[ver]) {
    if (!used[to]) {
      Dfs(to);
      Dsu_unite(ver, to, ver);
    }
  }
  for (const auto& ask : asks[ver]) {
    if (used[ask.first]) {
      ans[ask.second] = parent[Dsu_get(ask.first)];
    }
  }
}

void ProcessQueries(const vector<pair<size_t, size_t>>& edges,
                    const vector<pair<size_t, size_t>>& queries) {
  graph.resize(edges.size() + 1);
  used.assign(edges.size() + 1, false);
  used2.assign(edges.size() + 1, false);
  dist.assign(edges.size() + 1, 0);
  dsu.resize(edges.size() + 1);
  parent.resize(edges.size() + 1);
  asks.resize(edges.size() + 1);
  ans.resize(queries.size());

  for (const auto& edge : edges) {
    graph[edge.first].push_back(edge.second);
    graph[edge.second].push_back(edge.first);
  }

  for (size_t index = 0; index < queries.size(); ++index) {
    asks[queries[index].first].push_back({queries[index].second, index});
    asks[queries[index].second].push_back({queries[index].first, index});
  }

  DFS(0);
  Dfs(0);

  for (size_t index = 0; index < queries.size(); ++index) {
    std::cout << dist[queries[index].first] + dist[queries[index].second] - 2 * dist[ans[index]]
              << '\n';
  }
}

signed main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  size_t num_vertexes;
  size_t num_queries;
  std::cin >> num_vertexes;
  vector<pair<size_t, size_t>> edges(num_vertexes - 1);
  std::cin >> edges >> num_queries;
  vector<pair<size_t, size_t>> queries(num_queries);
  std::cin >> queries;

  ProcessQueries(edges, queries);
  std::cout.flush();
  return 0;
}
