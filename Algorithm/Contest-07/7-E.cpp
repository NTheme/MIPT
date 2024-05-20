/*
E (1 балл, с ревью). Мосты
Ограничение времени	1 секунда
Ограничение памяти	33Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Дан неориентированный (быть может несвязный) граф. Требуется найти все мосты в
нем.

Формат ввода
В первой строке входного файла два натуральных числа n и m (1 ≤ n ≤ 20000, 1 ≤ m
≤ 200000) –количество вершин и рёбер в графе соответственно. Далее в m строках
перечислены рёбра графа. Каждое ребро задается парой чисел – номерами начальной
и конечной вершин соответственно.

Формат вывода
Первая строка выходного файла должна содержать одно натуральное число b –
количество мостов в заданном графе. На следующей строке выведите b чисел –
номера ребер, которые являются мостами, в возрастающем порядке. Ребра нумеруются
с единицы в том порядке, в котором они заданы во входном файле.
*/

#include <algorithm>
#include <iostream>
#include <vector>

using std::pair;
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
  explicit Graph(size_t n, const vector<pair<size_t, size_t>>& edges);
  vector<size_t> GetBridges() const;

 private:
  struct VertexTime;
  struct GraphEdge;
  vector<vector<GraphEdge>> graph_;
  void FindBridges(vector<size_t>& bridges, std::vector<VertexTime>& time,
                  size_t vertex, size_t parent) const;
};

struct Graph::GraphEdge {
  size_t from;
  size_t to;
  size_t number;

  GraphEdge() : from(0), to(0), number(0) {}
  GraphEdge(size_t fromn, size_t ton, size_t numbern)
      : from(fromn), to(ton), number(numbern) {}

  GraphEdge Inverted() const;
};

Graph::GraphEdge Graph::GraphEdge::Inverted() const {
  return GraphEdge(to, from, number);
}

struct Graph::VertexTime {
  size_t in;
  size_t min;
  bool meet;

  VertexTime() : in(0), min(0), meet(false) {}
  void UpdateIn(size_t time) { in = std::min(in, time); }
  void UpdateMin(size_t time) { min = std::min(min, time); }
};

Graph::Graph(size_t n, const vector<pair<size_t, size_t>>& edges) : graph_(n) {
  for (size_t number = 0; number < edges.size(); ++number) {
    GraphEdge add =
        GraphEdge(edges[number].first - 1, edges[number].second - 1, number + 1);
    graph_[edges[number].first - 1].push_back(add);
    graph_[edges[number].second - 1].push_back(add.Inverted());
  }
}

void Graph::FindBridges(vector<size_t>& bridges, std::vector<VertexTime>& time,
                       size_t vertex, size_t parent) const {
  if (parent != graph_.size()) {
    time[vertex].min = time[vertex].in = time[parent].in + 1;
  }
  time[vertex].meet = true;

  bool meet_parent = true;
  for (const auto& edge : graph_[vertex]) {
    if (edge.to == parent && meet_parent) {
      meet_parent = false;
      continue;
    }
    if (time[edge.to].meet) {
      time[vertex].UpdateMin(time[edge.to].in);
      continue;
    }

    FindBridges(bridges, time, edge.to, vertex);
    time[vertex].UpdateMin(time[edge.to].min);
    if (time[vertex].in < time[edge.to].min) {
      bridges.push_back(edge.number);
    }
  }
}

vector<size_t> Graph::GetBridges() const {
  vector<VertexTime> time(graph_.size());

  vector<size_t> bridges;
  for (size_t number = 0; number < graph_.size(); ++number) {
    if (!time[number].meet) {
      FindBridges(bridges, time, number, graph_.size());
    }
  }
  std::sort(bridges.begin(), bridges.end());

  return bridges;
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t n, m;
  std::cin >> n >> m;
  vector<pair<size_t, size_t>> edges(m);
  std::cin >> edges;

  auto bridges = Graph(n, edges).GetBridges();
  std::cout << bridges.size() << '\n' << bridges << '\n';

  return 0;
}
