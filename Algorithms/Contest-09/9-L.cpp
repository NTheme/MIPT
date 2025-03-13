/*
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

*/

#include <algorithm>
#include <iostream>
#include <queue>
#include <vector>

using std::pair;
using std::queue;
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
    out << elem << ' ';
  }
  return out;
}

struct Edge {
  long long from = 0;
  long long to = 0;
  long long capacity = 0;
  long long index = 0;
  long long cps = 0;

  Edge() = default;
  Edge(long long from_n, long long to_n, long long capacity_n,
       long long index_n)
      : from(from_n),
        to(to_n),
        capacity(capacity_n),
        index(index_n),
        cps(capacity_n) {}
};

std::istream& operator>>(std::istream& in, Edge& edge) {
  in >> edge.from >> edge.to >> edge.capacity;
  --edge.from;
  --edge.to;
  return in;
}

vector<vector<long long>> graph;
vector<Edge> edges;
vector<long long> dist;
vector<long long> place;

bool BFS(long long start, long long end) {
  dist.assign(graph.size(), -1);
  queue<long long> que;
  dist[start] = 0;
  que.push(start);

  while (!que.empty()) {
    long long ver = que.front();
    que.pop();
    for (auto& index : graph[ver]) {
      auto edge = edges[index];
      if (dist[edge.to] == -1 && edge.capacity > 0) {
        dist[edge.to] = dist[ver] + 1;
        que.push(edge.to);
      }
    }
  }
  return dist[end] != -1;
}

long long DFS(long long ver, long long flow, long long end) {
  if (ver == end) {
    return flow;
  }

  for (; place[ver] < (long long)graph[ver].size(); ++place[ver]) {
    auto ind = graph[ver][place[ver]];
    if (dist[edges[ind].to] == dist[ver] + 1 && edges[ind].capacity > 0) {
      long long push =
          DFS(edges[ind].to, std::min(flow, edges[ind].capacity), end);
      if (push > 0) {
        edges[ind].capacity -= push;
        edges[ind ^ 1].capacity += push;
        return push;
      }
    }
  }
  return 0;
}

void Dinic(long long start, long long end) {
  while (true) {
    if (!BFS(start, end)) {
      break;
    }

    place.assign(graph.size(), 0);
    while (true) {
      long long push = DFS(start, 1e9, end);
      if (push == 0) {
        break;
      }
    }
  }
}

vector<bool> used;

void DFs(long long ver) {
  used[ver] = true;

  for (const auto& index : graph[ver]) {
    auto edge = edges[index];
    if (!used[edge.to] && edge.capacity > 0) {
      DFs(edge.to);
    }
  }
}

void Task() {
  long long num_ver;
  long long num_edges;
  std::cin >> num_ver >> num_edges;
  long long start = 0;
  long long end = num_ver - 1;

  graph.resize(num_ver);
  used.assign(num_ver, false);
  for (long long index = 0; index < num_edges; ++index) {
    Edge edge;
    std::cin >> edge;
    graph[edge.from].push_back(edges.size());
    edges.push_back(Edge(edge.from, edge.to, edge.capacity, index));
    graph[edge.to].push_back(edges.size());
    edges.push_back(Edge(edge.to, edge.from, edge.capacity, index));
  }
  Dinic(start, end);
  DFs(start);

  vector<long long> indedes;
  long long ret = 0;
  for (auto& edge : edges) {
    if (used[edge.from] && !used[edge.to]) {
      indedes.push_back(edge.index + 1);
      ret += edge.cps;
    }
  }

  std::cout << indedes.size() << ' ' << ret << '\n' << indedes << '\n';
}

signed main() {
  std::cin.tie(nullptr)->sync_with_stdio(false);
  std::cout.precision(20);
  Task();

  std::cout.flush();
  return 0;
}
