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

struct Road {
  size_t end;
  size_t number;

  Road() : end(0), number(0) {}
  Road(size_t end_n, size_t number_m) : end(end_n), number(number_m) {}
};

struct Vertex {
  size_t in;
  size_t min;
  bool meet;

  Vertex();
  void UpdateIn(size_t time);
  void UpdateMin(size_t time);
};

Vertex::Vertex() : in(0), min(0), meet(false) {}

void Vertex::UpdateIn(size_t time) { in = std::min(in, time); }

void Vertex::UpdateMin(size_t time) { min = std::min(min, time); }

void Dfs(const vector<vector<Road>>& graph, vector<size_t>& bridges,
         std::vector<Vertex>& time, size_t vertex, size_t parent) {
  if (parent != graph.size()) {
    time[vertex].min = time[vertex].in = time[parent].in + 1;
  }
  time[vertex].meet = true;

  bool meet_parent = true;
  for (const auto& edge : graph[vertex]) {
    if (edge.end == parent && meet_parent) {
      meet_parent = false;
      continue;
    }
    if (time[edge.end].meet) {
      time[vertex].UpdateMin(time[edge.end].in);
      continue;
    }

    Dfs(graph, bridges, time, edge.end, vertex);
    time[vertex].UpdateMin(time[edge.end].min);
    if (time[vertex].in < time[edge.end].min) {
      bridges.push_back(edge.number);
    }
  }
}

vector<size_t> FindBridges(const vector<vector<Road>>& graph) {
  vector<Vertex> time(graph.size());

  vector<size_t> bridges;
  for (size_t index = 0; index < graph.size(); ++index) {
    if (!time[index].meet) {
      Dfs(graph, bridges, time, index, graph.size());
    }
  }
  std::sort(bridges.begin(), bridges.end());

  return bridges;
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t n, m;
  std::cin >> n >> m;

  vector<vector<Road>> graph(n);
  for (size_t edge = 0; edge < m; ++edge) {
    size_t ver1, ver2;
    std::cin >> ver1 >> ver2;
    graph[--ver1].push_back(Road(--ver2, edge + 1));
    graph[ver2].push_back(Road(ver1, edge + 1));
  }

  auto bridges = FindBridges(graph);
  std::cout << bridges.size() << '\n' << bridges << '\n';

  return 0;
}
