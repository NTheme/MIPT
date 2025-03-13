/*

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

*/

#include <algorithm>
#include <chrono>
#include <iostream>
#include <set>
#include <vector>

using std::pair;
using std::set;
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

auto startt = std::chrono::high_resolution_clock::now();

namespace sf {
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

size_t Dsu_get(size_t ver) {
  return ver == dsu[ver] ? ver : dsu[ver] = Dsu_get(dsu[ver]);
}

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
                    const vector<pair<size_t, size_t>>& queries, size_t start,
                    size_t num_ver) {
  graph.resize(num_ver);
  used.assign(num_ver, false);
  used2.assign(num_ver, false);
  dist.assign(num_ver, 0);
  dsu.resize(num_ver);
  parent.resize(num_ver); 
  asks.resize(num_ver);
  ans.resize(queries.size());

  for (const auto& edge : edges) {
    graph[edge.first].push_back(edge.second);
    graph[edge.second].push_back(edge.first);
  }

  for (size_t index = 0; index < queries.size(); ++index) {
    asks[queries[index].first].push_back({queries[index].second, index});
    asks[queries[index].second].push_back({queries[index].first, index});
  }

  DFS(start);
  Dfs(start);

  for (size_t index = 0; index < queries.size(); ++index) {
    std::cout << dist[ans[index]] << '\n';
  }
}
}  // namespace sf

namespace c7 {
struct Vertex {
  size_t in;
  size_t min;
  bool meet;

  Vertex() : in(0), min(0), meet(false) {}

  void UpdateIn(size_t time);
  void UpdateMin(size_t time);
};

void Vertex::UpdateIn(size_t time) { in = std::min(in, time); }

void Vertex::UpdateMin(size_t time) { min = std::min(min, time); }

struct Edge {
  size_t start;
  size_t end;
  size_t number;
  bool* is_bridge;

  Edge() : start(0), end(0), number(0), is_bridge(nullptr) {}
  Edge(size_t start_n, size_t end_n, size_t number_n)
      : start(start_n), end(end_n), number(number_n), is_bridge(nullptr) {}
  Edge(size_t start_n, size_t end_n, size_t number_n, bool* is_bridge_n)
      : start(start_n), end(end_n), number(number_n), is_bridge(is_bridge_n) {}
};

void Dfs1(vector<vector<Edge>>& graph, std::vector<Vertex>& time, size_t vertex,
          size_t parent) {
  if (parent != graph.size()) {
    time[vertex].min = time[vertex].in = time[parent].in + 1;
  }
  time[vertex].meet = true;

  bool meet_parent = true;
  for (auto& edge : graph[vertex]) {
    if (edge.end == parent && meet_parent) {
      meet_parent = false;
      continue;
    }
    if (time[edge.end].meet) {
      time[vertex].UpdateMin(time[edge.end].in);
      continue;
    }

    Dfs1(graph, time, edge.end, vertex);
    time[vertex].UpdateMin(time[edge.end].min);
    if (time[vertex].in < time[edge.end].min) {
      *edge.is_bridge = true;
    }
  }
}

vector<size_t> Dfs2(const vector<vector<Edge>>& graph, vector<Vertex>& time,
                    size_t vertex) {
  time[vertex].meet = true;

  vector<size_t> component = {vertex};
  for (const auto& next : graph[vertex]) {
    if (!time[next.end].meet) {
      auto add = Dfs2(graph, time, next.end);
      copy(add.begin(), add.end(), back_inserter(component));
    }
  }

  return component;
}

pair<vector<pair<size_t, size_t>>, pair<vector<size_t>, size_t>> Get(
    const vector<pair<size_t, size_t>>& edges, size_t blllllll) {
  bool* bridges = new bool[edges.size()]{};
  vector<vector<Edge>> graph(blllllll);
  for (size_t edge = 0; edge < edges.size(); ++edge) {
    graph[edges[edge].first].push_back(
        Edge(edges[edge].first, edges[edge].second, edge, bridges + edge));
    graph[edges[edge].second].push_back(
        Edge(edges[edge].second, edges[edge].first, edge, bridges + edge));
  }
  vector<vector<Edge>> graph_new(graph.size());
  vector<Vertex> time(graph.size());
  for (size_t index = 0; index < graph.size(); ++index) {
    if (!time[index].meet) {
      Dfs1(graph, time, index, graph.size());
    }
  }
  time.assign(graph.size(), Vertex());
  for (const auto& vertex : graph) {
    for (const auto& edge : vertex) {
      if (!*edge.is_bridge) {
        graph_new[edge.start].push_back(edge);
      }
    }
  }
  size_t num = 0;
  vector<size_t> components(graph.size());
  for (size_t vertex = 0; vertex < graph.size(); vertex++) {
    if (!time[vertex].meet) {
      auto component = Dfs2(graph_new, time, vertex);
      for (const auto& vertex : component) {
        components[vertex] = num;
      }
      ++num;
    }
  }
  vector<bool> numed(edges.size());
  for (const auto& vertex : graph) {
    for (const auto& edge : vertex) {
      if (*edge.is_bridge) {
        numed[edge.number] = true;
      }
    }
  }
  vector<pair<size_t, size_t>> edg;
  for (size_t edge = 0; edge < edges.size(); ++edge) {
    if (numed[edge]) {
      edg.push_back(
          {components[edges[edge].first], components[edges[edge].second]});
    }
  }
  delete[] bridges;
  return {edg, {components, num}};
}
}  // namespace c7

void Task() {
  size_t num_vertexes;
  size_t num_edges;
  size_t num_queries;
  size_t end;
  std::cin >> num_vertexes >> num_edges >> end;
  vector<pair<size_t, size_t>> edges(num_edges);
  std::cin >> edges >> num_queries;
  vector<pair<size_t, size_t>> queries(num_queries);
  std::cin >> queries;
  startt = std::chrono::high_resolution_clock::now();

  auto graph = c7::Get(edges, num_vertexes);
  for (auto& p : graph.second.first) {
    if (p >= graph.second.second) {
      throw;
    }
  }
  if (graph.second.first.size() != num_vertexes) {
    throw;
  }

  vector<pair<size_t, size_t>> que;
  for (const auto& query : queries) {
    que.push_back(
        {graph.second.first[query.first], graph.second.first[query.second]});
  }

  sf::ProcessQueries(graph.first, que, graph.second.first[end - 1],
                     graph.second.second);
}

signed main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  Task();

  std::cout.flush();
  return 0;
}
