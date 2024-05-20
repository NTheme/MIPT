/*

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

*/

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <set>
#include <vector>

using std::array;
using std::pair;
using std::set;
using std::shared_ptr;
using std::string;
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
  size_t to;
  int capacity, cost;
  Edge* twin = nullptr;

  Edge() = default;
  Edge(size_t to_n, int capacity_n, int cost_n = 0)
      : to(to_n), capacity(capacity_n), cost(cost_n) {}

  static void Connect(Edge& edge1, Edge& edge2);
};

std::istream& operator>>(std::istream& inp, Edge& edge) {
  inp >> edge.cost;
  return inp;
}

void Edge::Connect(Edge& edge1, Edge& edge2) {
  edge1.twin = &edge2;
  edge2.twin = &edge1;
}

static const long long kInf = 2e9;
static const long long kMaxN = 602;

array<array<Edge, kMaxN>, kMaxN> graph;
array<int, kMaxN> dist, prev, price, flow;
array<bool, kMaxN> used;
size_t num_side;

void FB() {
  price.fill(kInf);
  price[0] = 0;

  for (size_t ind = 0; ind < num_side - 1; ++ind) {
    for (size_t ver = 0; ver < num_side; ++ver) {
      for (auto& edge : graph[ver]) {
        if (edge.twin == nullptr) {
          break;
        }
        if (edge.capacity > 0 && price[edge.to] > price[ver] + edge.cost) {
          price[edge.to] = price[ver] + edge.cost;
        }
      }
    }
  }
}

void BFS() {
  dist[0] = 0;
  set<pair<int, size_t>> queue;
  queue.insert({0, dist[0]});

  while (!queue.empty()) {
    size_t ind = (*queue.begin()).second;
    queue.erase(queue.begin());
    used[ind] = true;
    for (auto& edge : graph[ind]) {
      if (edge.twin == nullptr) {
        break;
      }
      if (!used[edge.to] && edge.capacity > 0 &&
          dist[edge.to] > dist[ind] + edge.cost + price[ind] - price[edge.to]) {
        queue.erase({dist[edge.to], edge.to});
        dist[edge.to] = dist[ind] + edge.cost + price[ind] - price[edge.to];
        flow[edge.to] = std::min(flow[edge.to], edge.capacity);
        prev[edge.to] = ind;
        queue.insert({dist[edge.to], edge.to});
      }
    }
  }
}

bool Dijkstra() {
  prev.fill(-1);
  flow.fill(kInf);
  dist.fill(kInf);
  used.fill(false);

  BFS();

  if (dist[num_side - 1] >= kInf) {
    return false;
  }

  size_t ind = num_side - 1;
  while (ind != 0) {
    for (auto& edge : graph[prev[ind]]) {
      if (edge.twin == nullptr) {
        break;
      }
      if (edge.to == ind) {
        edge.capacity -= flow[num_side - 1];
        edge.twin->capacity += flow[num_side - 1];
      }
    }
    ind = prev[ind];
  }
  return true;
}

void Task() {
  std::cin >> num_side;

  for (size_t num_x = 0; num_x < num_side; ++num_x) {
    for (size_t num_y = 0; num_y < num_side; ++num_y) {
      int cost;
      std::cin >> cost;
      graph[num_x + 1][num_y] = Edge(num_side + 1 + num_y, 1, cost);
      graph[num_side + 1 + num_y][num_x] = Edge(num_x + 1, 0, -cost);
      Edge::Connect(graph[num_x + 1][num_y], graph[num_side + 1 + num_y][num_x]);
    }
  }
  for (size_t ind = 1; ind <= num_side; ++ind) {
    graph[0][ind - 1] = Edge(ind, 1, 0);
    graph[ind][num_side] = Edge(0, 0, 0);
    Edge::Connect(graph[0][ind - 1], graph[ind][num_side]);
  }

  for (size_t ind = num_side + 1; ind <= 2 * num_side; ++ind) {
    graph[ind][num_side] = Edge(2 * num_side + 1, 1, 0);
    graph[2 * num_side + 1][ind - num_side - 1] = Edge(ind, 0, 0);
    Edge::Connect(graph[ind][num_side], graph[2 * num_side + 1][ind - num_side - 1]);
  }

  num_side = 2 * num_side + 2;

  FB();

  int cost = 0;
  while (Dijkstra()) {
    for (size_t ind = 0; ind < num_side; ++ind) {
      price[ind] += dist[ind];
    }
    cost += flow[num_side - 1] * price[num_side - 1];
  }

  std::cout << cost << '\n';
  for (size_t ind = 1; ind < num_side / 2; ++ind) {
    for (auto& edge : graph[ind]) {
      if (edge.twin == nullptr) {
        break;
      }
      if (edge.to >= num_side / 2 && edge.to < num_side - 1 && edge.capacity == 0) {
        std::cout << ind << ' ' << edge.to - num_side / 2 + 1 << '\n';
      }
    }
  }
}

signed main() {
  std::cin.tie(nullptr)->sync_with_stdio(false);
  std::cout.precision(20);

  Task();

  std::cout.flush();
  return 0;
}
