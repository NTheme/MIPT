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
using std::string;
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

static const long long kInf = 1e18;

vector<vector<long long>> capacity;
vector<long long> overflow;
vector<long long> height;

long long flow = 0;
queue<long long> que;

long long Push(long long from, long long to, long long start, long long end) {
  long long Push = std::min(overflow[from], capacity[from][to]);

  if (to == end) {
    flow += Push;
  }
  if (overflow[to] == 0 && to != end) {
    que.push(to);
  }
  overflow[from] -= Push;
  overflow[to] += Push;
  capacity[from][to] -= Push;
  capacity[to][from] += Push;
  return Push;
}

void Lift(long long ver) {
  long long min_ver = kInf;
  for (long long index = 0; index < (long long)capacity.size(); index++) {
    if (capacity[ver][index] > 0) {
      min_ver = std::min(min_ver, height[index]);
    }
  }
  if (min_ver != kInf) {
    height[ver] = min_ver + 1;
  }
}

void Discharge(long long ver, long long start, long long end) {
  while (overflow[ver] > 0) {
    for (long long index = 0; index < (long long)capacity.size(); index++) {
      if (capacity[ver][index] > 0 && height[ver] == height[index] + 1) {
        Push(ver, index, start, end);
      }
    }
    Lift(ver);
  }
}

struct Edge {
  long long from = 0;
  long long to = 0;
  long long capacity = 0;

  Edge() = default;
  Edge(long long from_n, long long to_n, long long capacity_n)
      : from(from_n), to(to_n), capacity(capacity_n) {}
};

std::istream& operator>>(std::istream& in, Edge& edge) {
  in >> edge.from >> edge.to >> edge.capacity;
  --edge.from;
  --edge.to;
  return in;
}

void Task() {
  long long nnn, mmm;
  std::cin >> nnn >> mmm;
  long long start = 0, end = nnn - 1;
  vector<Edge> edges(mmm);
  capacity.assign(nnn, vector<long long>(nnn));
  overflow.assign(nnn, 0);
  height.assign(nnn, 0);
  overflow[start] = kInf;
  height[start] = nnn;

  for (long long index = 0; index < mmm; ++index) {
    std::cin >> edges[index];
    capacity[edges[index].from][edges[index].to] += edges[index].capacity;
  }

  for (long long index = 0; index < nnn; index++) {
    if (capacity[start][index] > 0) {
      Push(start, index, start, end);
    }
  }

  while (!que.empty()) {
    long long ver = que.front();
    que.pop();
    Discharge(ver, start, end);
  }

  std::cout << flow << '\n';
  for (const auto& edge : edges) {
    long long flow_s =
        std::max(edge.capacity - capacity[edge.from][edge.to], (long long)0);
    capacity[edge.from][edge.to] -=
        std::min(capacity[edge.from][edge.to], edge.capacity);
    std::cout << flow_s << '\n';
  }
}

signed main() {
  std::cin.tie(nullptr)->sync_with_stdio(false);
  std::cout.precision(20);

  Task();

  std::cout.flush();
  return 0;
}
