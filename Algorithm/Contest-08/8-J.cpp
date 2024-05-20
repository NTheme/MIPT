/*

*/

#include <algorithm>
#include <iostream>
#include <queue>
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
  int to;
  int weight;

  Edge() : to(0), weight(0) {}
  Edge(int ton, int weightn) : to(ton), weight(weightn) {}
};

const int kVer = 1000000, kTel = 500, kAbracadabra = 1e9;

int GetDistance(const vector<Edge>* graph, int end) {
  vector<int> dist(kVer + kTel, kAbracadabra);
  std::priority_queue<pair<int, int>> queue;
  dist[0] = 0;
  queue.push({dist[0], 0});

  while (!queue.empty()) {
    int vertex = queue.top().second;
    int cur_d = -queue.top().first;
    if (vertex == end) {
      break;
    }
    queue.pop();
    if (cur_d > dist[vertex]) {
      continue;
    }
    for (const auto& edge : graph[vertex]) {
      if (dist[vertex] + edge.weight < dist[edge.to]) {
        dist[edge.to] = dist[vertex] + edge.weight;
        queue.push({-dist[edge.to], edge.to});
      }
    }
  }

  return dist[end];
}

void Task() {
  int num_ver;
  int num_tel;
  int g_up;
  int g_dn;
  int g_in;
  int g_jp;
  int nver;
  int cver;
  std::cin >> num_ver >> g_up >> g_dn >> g_in >> g_jp >> num_tel;

  static vector<Edge> graph[kVer + kTel];

  for (int ind = 1; ind < kVer; ++ind) {
    graph[ind - 1].push_back(Edge(ind, g_up));
    graph[ind].push_back(Edge(ind - 1, g_dn));
  }

  for (int ind = kVer; ind < kVer + num_tel; ++ind) {
    std::cin >> nver;
    for (int index = 0; index < nver; ++index) {
      std::cin >> cver;
      graph[cver - 1].push_back(Edge(ind, g_in));
      graph[ind].push_back(Edge(cver - 1, g_jp));
    }
  }

  std::cout << GetDistance(graph, num_ver - 1);
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);

  Task();

  return 0;
}
