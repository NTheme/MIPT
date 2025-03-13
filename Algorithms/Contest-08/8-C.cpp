/*

*/

#include <algorithm>
#include <array>
#include <iostream>
#include <queue>
#include <set>
#include <vector>

using std::pair;
using std::priority_queue;
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
  int from;
  int to;
  int weight;
  int length;

  Edge() : from(0), to(0), weight(0), length(0) {}
  Edge(int fromn, int ton, int weightn, int lengthn)
      : from(fromn), to(ton), weight(weightn), length(lengthn) {}

  Edge Inverted() const;
};

Edge Edge::Inverted() const { return Edge(to, from, weight, length); }

std::istream& operator>>(std::istream& inp, Edge& edge) {
  inp >> edge.from >> edge.to >> edge.weight >> edge.length;
  edge.from--;
  edge.to--;
  return inp;
}

class Graph {
 public:
  Graph(int sizen, int timen, const std::vector<Edge>& graphn);
  void GetDistance();

 private:
  const int kMaxDistance = 2e9;
  int size_;
  int time_;
  vector<vector<Edge>> graph_;
};

Graph::Graph(int sizen, int timen, const std::vector<Edge>& graphn)
    : size_(sizen), time_(timen), graph_(size_) {
  for (const auto& edge : graphn) {
    graph_[edge.from].push_back(edge);
    graph_[edge.to].push_back(edge.Inverted());
  }
}

void Graph::GetDistance() {
  vector<int> dist(size_ * (time_ + 1), kMaxDistance);
  vector<int> prev(size_ * (time_ + 1), size_ * (time_ + 1));

  const int kMjkdfnaskjf = 3;
  priority_queue<std::array<int, kMjkdfnaskjf>> queue;
  for (int tim = 0; tim <= time_; ++tim) {
    dist[(tim + 1) * size_ - 1] = 0;
    queue.push({-dist[(tim + 1) * size_ - 1], tim, size_ - 1});
  }
  while (!queue.empty()) {
    auto ver = queue.top();
    queue.pop();
    int cur = ver[1] * size_ + ver[2];
    if (ver[0] > dist[cur]) {
      continue;
    }

    for (const auto& edge : graph_[ver[2]]) {
      int to = (ver[1] - edge.length) * size_ + edge.to;
      if (to < 0) {
        continue;
      }
      if (dist[cur] + edge.weight < dist[to]) {
        dist[to] = dist[cur] + edge.weight; 
        prev[to] = cur;
        queue.push({-dist[to], ver[1] - edge.length, edge.to});
      }
    }
  }

  if (dist[0] == kMaxDistance) {
    std::cout << -1 << '\n';
    return;
  }
  vector<int> path;
  for (int ver = 0; ver != size_ * (time_ + 1); ver = prev[ver]) {
    path.push_back(ver % size_ + 1);
  }

  std::cout << dist[0] << '\n';
  std::cout << path.size() << '\n' << path << '\n';
}

void Task() {
  int num_vertexes;
  int num_edges;
  int time;

  std::cin >> num_vertexes >> num_edges >> time;
  std::vector<Edge> edges(num_edges);
  std::cin >> edges;
  Graph graph(num_vertexes, time, edges);

  graph.GetDistance();
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);

  Task();

  return 0;
}
