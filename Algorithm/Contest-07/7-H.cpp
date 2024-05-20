/*

*/

#include <algorithm>
#include <iostream>
#include <vector>

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

int main() {
  std::cin.tie(nullptr)->sync_with_stdio(false);
  std::cout.precision(20);
  size_t n, m, num = 0, count = 0;
  std::cin >> n >> m;
  bool* bridges = new bool[m]{};
  vector<vector<Edge>> graph(n), graph_new(n);
  for (size_t edge = 0; edge < m; ++edge) {
    size_t ver1, ver2;
    std::cin >> ver1 >> ver2;
    graph[ver1 - 1].push_back(
        Edge(ver1 - 1, ver2 - 1, edge + 1, bridges + edge));
    graph[ver2 - 1].push_back(
        Edge(ver2 - 1, ver1 - 1, edge + 1, bridges + edge));
  }
  vector<Vertex> time(graph.size());
  for (size_t index = 0; index < graph.size(); ++index) {
    if (!time[index].meet) {
      Dfs1(graph, time, index, graph.size());
    }
  }
  time.assign(n, Vertex());
  for (const auto& vertex : graph) {
    for (const auto& edge : vertex) {
      if (!*edge.is_bridge) {
        graph_new[edge.start].push_back(edge);
      }
    }
  }
  vector<size_t> components(n);
  for (size_t vertex = 0; vertex < n; vertex++) {
    if (!time[vertex].meet) {
      auto component = Dfs2(graph_new, time, vertex);
      for (const auto& vertex : component) {
        components[vertex] = num;
      }
      ++num;
    }
  }
  time.assign(n, Vertex());
  graph_new.assign(n, {});
  for (const auto& vertex : graph) {
    for (const auto& edge : vertex) {
      if (*edge.is_bridge) {
        Edge add = Edge(components[edge.start], components[edge.end], 0);
        graph_new[components[edge.start]].push_back(add);
      }
    }
  }
  for (const auto& vertex : graph_new) {
    if (vertex.size() == 1) {
      ++count;
    }
  }
  std::cout << (count + 1) / 2 << '\n';
  delete[] bridges;
  return 0;
}
