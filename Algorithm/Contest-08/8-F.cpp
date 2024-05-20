/*

*/

#include <algorithm>
#include <iostream>
#include <map>
#include <unordered_set>
#include <vector>

using std::map;
using std::pair;
using std::unordered_set;
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
};

std::istream& operator>>(std::istream& inp, Edge& edge) {
  inp >> edge.from >> edge.to >> edge.weight;
  return inp;
}

class Graph {
 public:
  Graph(size_t n, const std::vector<Edge>& edges);
  vector<size_t> GetFullMaximalPath(const std::vector<size_t>& path);

 private:
  const long long kMaxDistance = 1e18;
  const long long kMaxDist = 1e12;

  size_t size_;
  vector<vector<long long>> graph_;
  map<size_t, size_t> edg_;

  vector<size_t> GetMaxPath(size_t utex, size_t vtex,
                            const std::vector<std::vector<size_t>>& prev) const;
};

Graph::Graph(size_t n, const std::vector<Edge>& edges)
    : size_(n), graph_(n, vector<long long>(n, -kMaxDistance)) {
  for (size_t index = 0; index < n; ++index) {
    graph_[index][index] = 0;
  }
  for (const auto& edge : edges) {
    graph_[edge.from - 1][edge.to - 1] = edge.weight;
    edg_[(edge.from - 1) * size_ + edge.to - 1] = edg_.size() + 1;
  }
}

vector<size_t> Graph::GetMaxPath(
    size_t utex, size_t vtex,
    const std::vector<std::vector<size_t>>& prev) const {
  if (prev[utex][vtex] == size_) {
    return {utex, vtex};
  }
  auto left = GetMaxPath(utex, prev[utex][vtex], prev);
  auto right = GetMaxPath(prev[utex][vtex], vtex, prev);
  left.insert(left.end(), right.begin() + 1, right.end());
  return left;
}

vector<size_t> Graph::GetFullMaximalPath(const std::vector<size_t>& path) {
  vector<vector<long long>> dist = graph_;
  vector<vector<size_t>> prev(size_, vector<size_t>(size_, size_));
  for (size_t k = 0; k < graph_.size(); ++k) {
    for (size_t i = 0; i < graph_.size(); ++i) {
      for (size_t j = 0; j < graph_.size(); ++j) {
        if (dist[i][j] < dist[i][k] + dist[k][j]) {
          dist[i][j] = std::max(dist[i][j], dist[i][k] + dist[k][j]);
          prev[i][j] = k;
        }
      }
    }
  }

  vector<size_t> edpath;
  for (size_t town = 1; town < path.size(); ++town) {
    for (size_t index = 0; index < graph_.size(); ++index) {
      if (dist[index][index] > 0 &&
          dist[path[town - 1] - 1][index] > -kMaxDist &&
          dist[index][path[town] - 1] > -kMaxDist) {
        std::cout << "infinitely kind\n";
        exit(0);
      }
    }
    auto way = GetMaxPath(path[town - 1] - 1, path[town] - 1, prev);
    for (size_t index = 1; index < way.size(); ++index) {
      if (edg_.contains(way[index - 1] * size_ + way[index])) {
        edpath.push_back(edg_[way[index - 1] * size_ + way[index]]);
      }
    }
  }

  return edpath;
}

void Task() {
  size_t nnn;
  size_t mmm;
  size_t kkk;
  std::cin >> nnn >> mmm >> kkk;
  std::vector<Edge> edges(mmm);
  std::vector<size_t> path(kkk);
  std::cin >> edges >> path;
  Graph graph(nnn, edges);
  auto anss = graph.GetFullMaximalPath(path);
  std::cout << anss.size() << '\n' << anss << '\n';
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);

  Task();

  return 0;
}
