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
  int weight;

  Edge() : from(0), to(0), weight(0) {}
  Edge(size_t fromn, size_t ton, int weightn)
      : from(fromn), to(ton), weight(weightn) {}

  Edge Inverted() const;
};

Edge Edge::Inverted() const { return Edge(to, from, weight); }

std::istream& operator>>(std::istream& inp, Edge& edge) {
  inp >> edge.from >> edge.to >> edge.weight;
  return inp;
}

class Graph {
 public:
  Graph(size_t n, const std::vector<Edge>& edgesn);

  vector<int> GetDistances(size_t start);

 private:
  const int kMaxDistance = 30000;

  size_t size_;
  vector<Edge> edges_;
};

Graph::Graph(size_t sizen, const std::vector<Edge>& edgesn) : size_(sizen) {
  for (const auto& edge : edgesn) {
    edges_.push_back(Edge(edge.from - 1, edge.to - 1, edge.weight));
  }
}

vector<int> Graph::GetDistances(size_t start) {
  vector<int> dist(size_, kMaxDistance);
  dist[start] = 0;
  for (size_t i = 0; i < size_; ++i) {
    for (size_t j = 0; j < edges_.size(); ++j) {
      if (dist[edges_[j].from] < kMaxDistance) {
        dist[edges_[j].to] = std::min(dist[edges_[j].to],
                                      dist[edges_[j].from] + edges_[j].weight);
      }
    }
  }
  return dist;
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);

  size_t num_vertexes;
  size_t num_edges;

  std::cin >> num_vertexes >> num_edges;
  std::vector<Edge> edges(num_edges);
  std::cin >> edges;
  std::cout << Graph(num_vertexes, edges).GetDistances(0);

  return 0;
}
