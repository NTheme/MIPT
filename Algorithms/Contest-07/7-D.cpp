/*
D (2 балла, с ревью). Компоненты сильной связности

Ограничение времени	1 секунда
Ограничение памяти	128Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Вам задан ориентированный граф с N вершинами и M ребрами (1 ≤ N ≤ 20000, 1 ≤ M ≤
200000). Найдите компоненты сильной связности заданного графа и топологически
отсортируйте его конденсацию.

Формат ввода
Граф задан во входном файле следующим образом: первая строка содержит числа N и
M. Каждая из следующих M строк содержит описание ребра — два целых числа из
диапазона от 1 до N — номера начала и конца ребра.

Формат вывода
На первой строке выведите число K — количество компонент сильной связности в
заданном графе. На следующей строке выведите N чисел — для каждой вершины
выведите номер компоненты сильной связности, которой принадлежит эта вершина.
Компоненты сильной связности должны быть занумерованы таким образом, чтобы для
любого ребра номер компоненты сильной связности его начала не превышал номера
компоненты сильной связности его конца.
*/

#include <algorithm>
#include <iostream>
#include <vector>

using std::pair;
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

class Graph {
 public:
  Graph(size_t n, const vector<pair<size_t, size_t>>& edges);

  vector<size_t> GetTopSorted() const;
  Graph GetInversedGraph() const;
  pair<size_t, vector<size_t>> GetStrongComponents() const;

 private:
  vector<vector<size_t>> graph_;

  void MakeTopSorted(vector<bool>& meet, vector<size_t>& order,
                     size_t vertex) const;
  vector<size_t> FindStrongComponents(vector<bool>& meet, size_t vertex) const;
};

Graph::Graph(size_t n, const vector<pair<size_t, size_t>>& edges) : graph_(n) {
  for (const auto& edge : edges) {
    graph_[edge.first - 1].push_back(edge.second - 1);
  }
}

void Graph::MakeTopSorted(vector<bool>& meet, vector<size_t>& order,
                          size_t vertex) const {
  meet[vertex] = true;
  for (const auto& next : graph_[vertex]) {
    if (!meet[next]) {
      MakeTopSorted(meet, order, next);
    }
  }
  order.push_back(vertex);
}

vector<size_t> Graph::FindStrongComponents(vector<bool>& meet,
                                           size_t vertex) const {
  meet[vertex] = true;
  vector<size_t> component = {vertex};
  for (const auto& next : graph_[vertex]) {
    if (!meet[next]) {
      auto add = FindStrongComponents(meet, next);
      copy(add.begin(), add.end(), back_inserter(component));
    }
  }
  return component;
}

vector<size_t> Graph::GetTopSorted() const {
  vector<size_t> order;
  vector<bool> meet(graph_.size());
  for (size_t index = 0; index < graph_.size(); ++index) {
    if (!meet[index]) {
      MakeTopSorted(meet, order, index);
    }
  }
  std::reverse(order.begin(), order.end());
  return order;
}

Graph Graph::GetInversedGraph() const {
  vector<pair<size_t, size_t>> edges;
  for (size_t vertex = 0; vertex < graph_.size(); ++vertex) {
    for (const auto& edge : graph_[vertex]) {
      edges.push_back({edge + 1, vertex + 1});
    }
  }
  return Graph(graph_.size(), edges);
}

pair<size_t, vector<size_t>> Graph::GetStrongComponents() const {
  vector<vector<size_t>> components;
  vector<size_t> list(graph_.size());

  auto order = GetTopSorted();
  auto graph_inv = GetInversedGraph();

  vector<bool> meet(graph_.size());
  for (size_t index = 0; index < graph_.size(); index++) {
    if (!meet[order[index]]) {
      components.push_back(graph_inv.FindStrongComponents(meet, order[index]));
      for (const auto& vertex : components.back()) {
        list[vertex] = components.size();
      }
    }
  }
  return {components.size(), list};
}

int main() {
  std::cin.tie(nullptr)->sync_with_stdio(false);
  std::cout.precision(20);

  size_t n, m;
  std::cin >> n >> m;
  vector<pair<size_t, size_t>> graph(m);
  for (auto& edge : graph) {
    std::cin >> edge.first >> edge.second;
  }

  auto strong_components = Graph(n, graph).GetStrongComponents();
  std::cout << strong_components.first << '\n'
            << strong_components.second << '\n';

  return 0;
}
