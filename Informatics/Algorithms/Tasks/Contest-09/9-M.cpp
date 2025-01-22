/*
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

*/

#include <algorithm>
#include <iostream>
#include <memory>
#include <queue>
#include <vector>

using std::pair;
using std::queue;
using std::shared_ptr;
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
  out << pair.first << ' ' << pair.second << '\n';
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
    out << elem;
  }
  return out;
}

struct Edge {
  long long to = 0;
  long long capacity = 0;
  shared_ptr<Edge> twin;
  bool straight = true;

  Edge() = default;
  Edge(long long to_n, long long capacity_n);

  static void Connect(shared_ptr<Edge>& edge1, shared_ptr<Edge>& edge2);
};

Edge::Edge(long long to_n, long long capacity_n)
    : to(to_n), capacity(capacity_n), twin(nullptr), straight(capacity != 0) {}

void Edge::Connect(shared_ptr<Edge>& edge1, shared_ptr<Edge>& edge2) {
  edge1->twin = edge2;
  edge2->twin = edge1;
}

static const long long kInf = (1 << 30);

vector<vector<shared_ptr<Edge>>> graph;
vector<bool> used;

bool DFS(long long ver, long long flow, long long end) {
  if (ver == end) {
    return true;
  }
  used[ver] = true;
  for (auto edge : graph[ver]) {
    if (!used[edge->to] && edge->capacity >= flow) {
      if (DFS(edge->to, flow, end)) {
        edge->capacity -= flow;
        edge->twin->capacity += flow;
        return true;
      }
    }
  }
  return false;
}

void DFs(long long ver) {
  used[ver] = true;
  for (auto edge : graph[ver]) {
    if (!used[edge->to] && edge->capacity > 0) {
      DFs(edge->to);
    }
  }
}

void Task() {
  long long num_x, num_y, num_blocked, num_allowed;
  std::cin >> num_x >> num_y >> num_blocked >> num_allowed;
  vector<vector<bool>> blocked(num_x, vector<bool>(num_y)),
      allowed(num_x, vector<bool>(num_y));
  graph.resize(num_x * num_y * 2);

  for (long long ind = 0; ind < num_blocked; ++ind) {
    pair<long long, long long> point;
    std::cin >> point;
    blocked[point.first][point.second] = true;
  }
  for (long long ind = 0; ind < num_allowed; ++ind) {
    pair<long long, long long> point;
    std::cin >> point;

    allowed[point.first][point.second] = true;

    auto edge1 = std::make_shared<Edge>(
        num_x * num_y + point.first * num_y + point.second, 1);
    auto edge2 = std::make_shared<Edge>(point.first * num_y + point.second, 0);
    Edge::Connect(edge1, edge2);

    graph[point.first * num_y + point.second].push_back(edge1);
    graph[num_x * num_y + point.first * num_y + point.second].push_back(edge2);
  }

  for (long long x_ind = 0; x_ind < num_x; ++x_ind) {
    for (long long y_ind = 0; y_ind < num_y; ++y_ind) {
      if (!blocked[x_ind][y_ind] && !allowed[x_ind][y_ind]) {
        auto edge1 =
            std::make_shared<Edge>(num_x * num_y + x_ind * num_y + y_ind, kInf);
        auto edge2 = std::make_shared<Edge>(x_ind * num_y + y_ind, 0);
        Edge::Connect(edge1, edge2);

        graph[x_ind * num_y + y_ind].push_back(edge1);
        graph[num_x * num_y + x_ind * num_y + y_ind].push_back(edge2);
      }
      if (x_ind > 0) {
        auto edge1 = std::make_shared<Edge>((x_ind - 1) * num_y + y_ind, kInf);
        auto edge2 =
            std::make_shared<Edge>(num_x * num_y + x_ind * num_y + y_ind, 0);
        Edge::Connect(edge1, edge2);

        graph[num_x * num_y + x_ind * num_y + y_ind].push_back(edge1);
        graph[(x_ind - 1) * num_y + y_ind].push_back(edge2);
      }
      if (x_ind < num_x - 1) {
        auto edge1 = std::make_shared<Edge>((x_ind + 1) * num_y + y_ind, kInf);
        auto edge2 =
            std::make_shared<Edge>(num_x * num_y + x_ind * num_y + y_ind, 0);
        Edge::Connect(edge1, edge2);

        graph[num_x * num_y + x_ind * num_y + y_ind].push_back(edge1);
        graph[(x_ind + 1) * num_y + y_ind].push_back(edge2);
      }
      if (y_ind > 0) {
        auto edge1 = std::make_shared<Edge>(x_ind * num_y + y_ind - 1, kInf);
        auto edge2 =
            std::make_shared<Edge>(num_x * num_y + x_ind * num_y + y_ind, 0);
        Edge::Connect(edge1, edge2);

        graph[num_x * num_y + x_ind * num_y + y_ind].push_back(edge1);
        graph[x_ind * num_y + y_ind - 1].push_back(edge2);
      }
      if (y_ind < num_y - 1) {
        auto edge1 = std::make_shared<Edge>(x_ind * num_y + y_ind + 1, kInf);
        auto edge2 =
            std::make_shared<Edge>(num_x * num_y + x_ind * num_y + y_ind, 0);
        Edge::Connect(edge1, edge2);

        graph[num_x * num_y + x_ind * num_y + y_ind].push_back(edge1);
        graph[x_ind * num_y + y_ind + 1].push_back(edge2);
      }
    }
  }

  pair<long long, long long> start_inp, end_inp;
  std::cin >> start_inp >> end_inp;
  long long start = start_inp.first * num_y + start_inp.second,
            end = num_x * num_y + end_inp.first * num_y + end_inp.second;

  long long blocks = 0;
  for (long long flow = kInf; flow > 0; flow /= 2) {
    used.assign(num_x * num_y * 2, false);
    while (DFS(start, flow, end)) {
      blocks += flow;
      used.assign(num_x * num_y * 2, false);
    }
  }
  if (blocks >= kInf) {
    std::cout << -1;
    return;
  }

  used.assign(num_x * num_y * 2, false);
  DFs(start);
  vector<long long> cells;
  for (long long ind = 0; ind < num_y * num_x * 2; ++ind) {
    if (used[ind]) {
      for (auto edge : graph[ind]) {
        if (edge->straight && !used[edge->to]) {
          cells.push_back(edge->to - num_x * num_y);
        }
      }
    }
  }
  std::cout << cells.size() << '\n';
  for (long long ind : cells) {
    std::cout << ind / num_y + 1 << " " << ind % num_y + 1 << '\n';
  }
}

signed main() {
  std::cin.tie(nullptr)->sync_with_stdio(false);
  std::cout.precision(20);

  Task();

  std::cout.flush();
  return 0;
}
