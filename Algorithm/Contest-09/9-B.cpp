/*

B (1 балл, с ревью). MST

Ограничение времени	0.45 секунд
Ограничение памяти	64Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Требуется найти в связном графе остовное дерево минимального веса.

Формат ввода
Первая строка входного файла содержит два натуральных числа n и m – количество
вершин и ребер графа соответственно (1 ≤ n ≤ 10^5, 0 ≤ m ≤ 10^5). Следующие m
строк содержат описание ребер по одному на строке. Ребро номер i описывается
тремя натуральными числами bi, ei и wi — номерами концов ребра и его вес
соответственно (1 ≤ bi, ei ≤ n, 0 ≤ wi ≤ 10^5).
Гарантируется, что граф связный.

Формат вывода
Выведите единственное число – вес минимального остовного дерева.

*/

#include <algorithm>
#include <compare>
#include <iostream>
#include <vector>

using std::pair;
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

class DSU {
 public:
  explicit DSU(size_t size);

  size_t Find(size_t elem);
  void Union(size_t elem1, size_t elem2);

 private:
  vector<size_t> parent_;
  vector<size_t> rank_;
};

DSU::DSU(size_t size) : parent_(size), rank_(size) {
  for (size_t index = 0; index < size; ++index) {
    parent_[index] = index;
  }
}

size_t DSU::Find(size_t elem) {
  if (elem == parent_[elem]) {
    return elem;
  }
  return parent_[elem] = Find(parent_[elem]);
}

void DSU::Union(size_t elem1, size_t elem2) {
  elem1 = Find(elem1);
  elem2 = Find(elem2);
  if (elem1 != elem2) {
    if (rank_[elem1] < rank_[elem2]) {
      std::swap(elem1, elem2);
    }
    parent_[elem2] = elem1;
    if (rank_[elem1] == rank_[elem2]) {
      ++rank_[elem1];
    }
  }
}

class Graph {
 public:
  struct Edge;

  explicit Graph(size_t num_vertexes);

  void AddEdge(const Edge& edge);
  size_t FindSpanTreeWeight();

 private:
  size_t m_num_vertexes;
  vector<Edge> m_edges;
};

struct Graph::Edge {
  size_t ver1 = 0;
  size_t ver2 = 0;
  size_t weight = 0;

  Edge() = default;
  std::strong_ordering operator<=>(const Edge& other) const;
};

std::strong_ordering Graph::Edge::operator<=>(const Edge& other) const {
  if (weight != other.weight) {
    return weight <=> other.weight;
  }
  if (ver1 != other.ver1) {
    return ver1 <=> other.ver1;
  }
  if (ver2 != other.ver2) {
    return ver2 <=> other.ver2;
  }
  return std::strong_ordering::equal;
}

std::istream& operator>>(std::istream& inp, Graph::Edge& edge) {
  inp >> edge.ver1 >> edge.ver2 >> edge.weight;
  --edge.ver1;
  --edge.ver2;
  return inp;
}

Graph::Graph(size_t num_vertexes) : m_num_vertexes(num_vertexes) {}

void Graph::AddEdge(const Edge& edge) { m_edges.push_back(edge); }

size_t Graph::FindSpanTreeWeight() {
  DSU set(m_num_vertexes);
  std::sort(m_edges.begin(), m_edges.end());

  size_t cost = 0;
  for (const auto& edge : m_edges) {
    if (set.Find(edge.ver1) != set.Find(edge.ver2)) {
      cost += edge.weight;
      set.Union(edge.ver1, edge.ver2);
    }
  }
  return cost;
}

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  size_t num_vertexes;
  size_t num_edges;
  std::cin >> num_vertexes >> num_edges;

  Graph graph(num_vertexes);
  for (size_t index = 0; index < num_edges; ++index) {
    Graph::Edge edge;
    std::cin >> edge;
    graph.AddEdge(edge);
  }

  std::cout << graph.FindSpanTreeWeight();
  std::cout.flush();
  return 0;
}
