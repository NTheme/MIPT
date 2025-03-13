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

struct Edge {
  long long to = 0;
  long long capacity = 0;

  Edge() = default;
  Edge(long long to_n, long long capacity_n) : to(to_n), capacity(capacity_n) {}
};

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
      long long push = DFS(edges[ind].to, std::min(flow, edges[ind].capacity), end);
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

void Task() {
  long long num_ver, max_win;
  std::cin >> num_ver >> max_win;

  graph.resize(num_ver + 1);
  for (long long ind = 1; ind < num_ver; ++ind) {
    long long p;
    std::cin >> p;
    graph[0].push_back(edges.size());
    edges.emplace_back(ind, p);
    graph[ind].push_back(edges.size());
    edges.emplace_back(0, 0);
  }

  long long add = 0;
  std::cin >> add;
  max_win += add;
  for (long long ind = 1; ind < num_ver; ++ind) {
    std::cin >> add;
  }

  for (long long ind = 1; ind < num_ver; ++ind) {
    graph[ind].push_back(edges.size());
    edges.emplace_back(num_ver, max_win);
    graph[num_ver].push_back(edges.size());
    edges.emplace_back(ind, 0);
  }

  for (long long x_cor = 0; x_cor < num_ver; ++x_cor) {
    for (long long y_cor = 0; y_cor < num_ver; ++y_cor) {
      std::cin >> add;
      if (x_cor == 0 || y_cor <= x_cor) {
        continue;
      }
      long long cur = graph.size();
      graph.push_back({});

      graph[0].push_back(edges.size());
      edges.emplace_back(cur, add);
      graph[cur].push_back(edges.size());
      edges.emplace_back(0, 0);

      graph[cur].push_back(edges.size());
      edges.emplace_back(x_cor, add);
      graph[x_cor].push_back(edges.size());
      edges.emplace_back(cur, 0);
      graph[cur].push_back(edges.size());
      edges.emplace_back(y_cor, add);
      graph[y_cor].push_back(edges.size());
      edges.emplace_back(cur, 0);
    }
  }

  Dinic(0, num_ver);

  for (auto& ind : graph[0]) {
    if (edges[ind].capacity > 0) {
      std::cout << "NO\n";
      return;
    }
  }
  std::cout << "YES\n";
}

signed main() {
  std::cin.tie(nullptr)->sync_with_stdio(false);
  std::cout.precision(20);
  Task();

  std::cout.flush();
  return 0;
}
