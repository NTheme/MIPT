/*
E (2 балла, с ревью). Циклический саботаж

Ограничение времени	0.1 секунда
Ограничение памяти	16Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Вы являетесь членом команды на космическом корабле. Ваша задача — убедиться, что
все системы функционируют должным образом и корабль может выполнить свою миссию.
Однако вы получили сообщение от подозрительного члена экипажа, что в одной из
систем есть саботаж.
Как член команды, вы знаете, что саботаж может негативно сказаться на миссии
корабля. В командном центре корабля есть компьютер, который показывает общую
производительность корабля.
Вам нужно написать программу, которая проверяет, есть ли саботаж в какой-либо из
систем корабля, который может повлиять на миссию корабля. Вы можете представить
системы корабля в виде ориентированного графа, где каждое ребро имеет вес.
Отрицательный вес означает наличие (и степень) саботажа в этой системе, а
положительный вес означает работоспособность (и надежность) системы.
Если в системе корабля найдется цикл отрицательного веса, то саботаж успешен.

Формат ввода
В первой строке содержится число N (1 ≤ N ≤ 100) — количество вершин графа
системы. В следующих N строках находится по N чисел — матрица смежности графа.
Веса ребер по модулю меньше 100000. Если ребра нет, соответствующее значение
равно 100000.

Формат вывода
В первой строке выведите YES, если цикл существует, или NO, в противном случае.
При наличии цикла выведите во второй строке количество вершин в нем (считая
одинаковые — первую и последнюю), а в третьей строке — вершины, входящие в этот
цикл, в порядке обхода. Если циклов несколько, то выведите любой из них.
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
  explicit Graph(const std::vector<std::vector<int>>& matrix);

  vector<size_t> GetComponents() const;
  vector<size_t> FindNegativeCycle() const;

 private:
  const int kMaxDistance = 100000;

  size_t size_;
  vector<Edge> edges_;
  void FindComponents(size_t vertex, const vector<vector<size_t>>& graph,
                      vector<bool>& used) const;
};

Graph::Graph(const std::vector<std::vector<int>>& matrix)
    : size_(matrix.size()) {
  for (size_t from = 0; from < matrix.size(); ++from) {
    for (size_t to = 0; to < matrix.size(); ++to) {
      if (matrix[from][to] != kMaxDistance) {
        edges_.push_back({to, from, matrix[from][to]});
      }
    }
  }
}

void Graph::FindComponents(size_t vertex, const vector<vector<size_t>>& graph,
                           vector<bool>& used) const {
  used[vertex] = true;
  for (const auto& to : graph[vertex]) {
    if (!used[to]) {
      FindComponents(to, graph, used);
    }
  }
}

vector<size_t> Graph::GetComponents() const {
  vector<size_t> comp;
  vector<bool> used(size_, false);

  vector<vector<size_t>> graph(size_);
  for (const auto& edge : edges_) {
    graph[edge.from].push_back(edge.to);
  }

  for (size_t index = 0; index < size_; ++index) {
    if (!used[index]) {
      FindComponents(index, graph, used);
      comp.push_back(index);
    }
  }
  return comp;
}

vector<size_t> Graph::FindNegativeCycle() const {
  vector<size_t> cycle;

  auto components = GetComponents();
  for (const auto& start : components) {
    vector<int> dist(size_, kMaxDistance);
    vector<size_t> prev(size_, size_);
    dist[start] = 0;

    size_t last_cycle = size_;
    for (size_t iter = 0; iter < size_; ++iter) {
      last_cycle = size_;
      for (size_t index = 0; index < edges_.size(); ++index) {
        if (dist[edges_[index].from] < kMaxDistance &&
            dist[edges_[index].to] >
                dist[edges_[index].from] + edges_[index].weight) {
          dist[edges_[index].to] =
              dist[edges_[index].from] + edges_[index].weight;
          prev[edges_[index].to] = edges_[index].from;
          last_cycle = edges_[index].to;
        }
      }
    }

    if (last_cycle != size_) {
      for (size_t iter = 0; iter < size_; ++iter) {
        last_cycle = prev[last_cycle];
      }
      for (size_t cur = last_cycle; cur != last_cycle || cycle.size() <= 1;
           cur = prev[cur]) {
        cycle.push_back(cur + 1);
      }
      cycle.push_back(last_cycle + 1);
      return cycle;
    }
  }
  return cycle;
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  size_t num_vertexes;
  std::cin >> num_vertexes;
  vector<vector<int>> matrix(num_vertexes, vector<int>(num_vertexes));
  for (auto& row : matrix) {
    std::cin >> row;
  }
  auto cycle = Graph(matrix).FindNegativeCycle();
  if (cycle.empty()) {
    std::cout << "NO\n";
  } else {
    std::cout << "YES\n" << cycle.size() << '\n' << cycle;
  }
  return 0;
}
