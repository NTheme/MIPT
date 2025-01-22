/*
A (1 балл, с ревью). Подлые карты

Ограничение времени	1 секунда
Ограничение памяти	64Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

В этой задаче вы являетесь членом команды на космическом корабле в игре "Among
Us". На корабле имеется N комнат, каждая из которых представлена вершиной в
графе. Вы и ваши командные товарищи пытаетесь выполнять задания и поддерживать
работоспособность корабля, но среди вас есть подлец, который пытается помешать
вам.
Подлец взломал навигационную систему корабля и создал K графов, представляющих
различные карты расположения комнат на корабле. Ваша задача состоит в том, чтобы
найти кратчайший путь от вашего текущего местоположения до каждой из других
комнат.
Каждая вершина представляет комнату на корабле, а каждое ребро представляет
коридор, соединяющий две соседние комнаты. Вес каждого ребра представляет время,
необходимое для перемещения между комнатами.
Сможете ли вы и ваши товарищи по команде пройти по кораблю и выполнить свои
задания?

Формат ввода
В первой строке входных данных задано число K — количество различных карт комнат
на корабле, где герои могут находиться. Далее следуют K блоков, каждый из
которых имеет следующую структуру.
Первая строка блока содержит два числа N и M, разделенные пробелом — количество
комнат и переходов. Далее следуют M строк, каждая из которых содержит по три
целых числа, разделенные пробелами. Первые два из них в пределах от 0 до N - 1
каждое и обозначают комнаты на концах соответствующего перехода, третье — в
пределах от 0 до 20000 и обозначает длину этого коридора. Далее, в последней
строке блока, записанное единственное число от 0 до N - 1 — вершина, где вы
расположены.
Количество различных карт в одном тесте K не превышает 5. Количество вершин не
превышает 60000, рёбер — 200000.

Формат вывода
Выведите в стандартный вывод K строк, в каждой из которых по Ni чисел,
разделенных пробелами — расстояния от указанной начальной комнаты до его 0-й,
1-й, 2-й и т. д. комнат (допускается лишний пробел после последнего числа). Если
некоторая комната недостижима от указанной начальной, вместо расстояния выводите
число 2009000999 (гарантировано, что все реальные расстояния меньше).
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
  explicit Graph(size_t n, const std::vector<Edge>& graphn);

  vector<int> GetDistances(size_t start) const;

 private:
  const size_t kMaxDistance = 2009000999;

  vector<vector<Edge>> graph_;
};

Graph::Graph(size_t n, const std::vector<Edge>& graphn) : graph_(n) {
  for (const auto& edge : graphn) {
    graph_[edge.from].push_back(edge);
    graph_[edge.to].push_back(edge.Inverted());
  }
}

vector<int> Graph::GetDistances(size_t start) const {
  vector<int> dist(graph_.size(), kMaxDistance);
  dist[start] = 0;
  set<pair<int, size_t>> queue;
  queue.insert({dist[start], start});
  while (!queue.empty()) {
    int vertex = queue.begin()->second;
    queue.erase(queue.begin());

    for (const auto& edge : graph_[vertex]) {
      if (dist[vertex] + edge.weight < dist[edge.to]) {
        queue.erase({dist[edge.to], edge.to});
        dist[edge.to] = dist[vertex] + edge.weight;
        queue.insert({dist[edge.to], edge.to});
      }
    }
  }

  return dist;
}

void Test() {}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  size_t num_tests;
  std::cin >> num_tests;
  while (num_tests-- > 0) {
    size_t num_vertexes;
    size_t num_edges;
    size_t start;
    std::cin >> num_vertexes >> num_edges;
    std::vector<Edge> edges(num_edges);
    std::cin >> edges >> start;
    std::cout << Graph(num_vertexes, edges).GetDistances(start);
  }
  return 0;
}
