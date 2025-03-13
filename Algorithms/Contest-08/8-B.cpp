/*

*/

#include <algorithm>
#include <iostream>
#include <set>
#include <vector>

using std::pair;
using std::set;
using std::vector;

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
  size_t from;
  size_t to;
  long long weight;

  Edge() : from(0), to(0), weight(0) {}
  Edge(size_t fromn, size_t ton, long long weightn)
      : from(fromn), to(ton), weight(weightn) {}

  Edge Inverted() const;
};

Edge Edge::Inverted() const { return Edge(to, from, weight); }

std::istream& operator>>(std::istream& inp, Edge& edge) {
  inp >> edge.from >> edge.to >> edge.weight;
  edge.from--;
  edge.to--;
  return inp;
}

class Graph {
 public:
  Graph(size_t n, const std::vector<Edge>& graphn);
  long long GetDistance(const vector<size_t>& start, size_t str, size_t end);

 private:
  const long long kMaxDistance = 2e18;
  vector<vector<Edge>> graph_;
};

Graph::Graph(size_t n, const std::vector<Edge>& graphn) : graph_(n) {
  for (const auto& edge : graphn) {
    graph_[edge.from].push_back(edge);
    graph_[edge.to].push_back(edge.Inverted());
  }
}
long long Graph::GetDistance(const vector<size_t>& start, size_t str,
                              size_t end) {
  vector<long long> dist(graph_.size(), kMaxDistance);
  set<pair<long long, size_t>> queue;
  for (const auto& ver : start) {
    dist[ver - 1] = 0;
    queue.insert({dist[ver - 1], ver - 1});
  }
  while (!queue.empty()) {
    size_t vertex = queue.begin()->second;
    queue.erase(queue.begin());

    for (const auto& edge : graph_[vertex]) {
      if (dist[vertex] + edge.weight < dist[edge.to]) {
        queue.erase({dist[edge.to], edge.to});
        dist[edge.to] = dist[vertex] + edge.weight;
        queue.insert({dist[edge.to], edge.to});
      }
    }
  }

  vector<long long> dist2(graph_.size(), kMaxDistance);
  dist2[str - 1] = 0;
  queue.insert({dist2[str - 1], str - 1});
  while (!queue.empty()) {
    size_t vertex = queue.begin()->second;
    queue.erase(queue.begin());

    for (const auto& edge : graph_[vertex]) {
      if (dist2[vertex] + edge.weight < dist2[edge.to] &&
          dist2[vertex] + edge.weight < dist[edge.to]) {
        queue.erase({dist2[edge.to], edge.to});
        dist2[edge.to] = dist2[vertex] + edge.weight;
        queue.insert({dist2[edge.to], edge.to});
      }
    }
  }

  return dist2[end - 1] == kMaxDistance ? -1 : dist2[end - 1];
}

void Task() {
  size_t num_vertexes;
  size_t num_edges;
  size_t num_vir;
  size_t start;
  size_t end;

  std::cin >> num_vertexes >> num_edges >> num_vir;
  std::vector<size_t> vir(num_vir);
  std::vector<Edge> edges(num_edges);
  std::cin >> vir >> edges >> start >> end;
  Graph graph(num_vertexes, edges);

  std::cout << graph.GetDistance(vir, start, end);
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);

  Task();

  return 0;
}
