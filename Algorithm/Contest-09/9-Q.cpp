/*

Q (4 балла, с ревью). Замещение

Ограничение времени	2 секунды
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

2022 год. Задан ориентированный граф поставок между N городами. Каждое ребро
обладает пропускной способностью и стоимостью перевозки единицы груза по нему.
Найдите максимальный поток из города с номером 1 в город под номером N. Среди
всех возможных вариантов выберите тот, который требует минимальной суммарной
стоимости. Формат ввода Первая строка входного файла содержит N и M — количество
городов и маршрутов поставок (2 ≤ N ≤ 100, 0 ≤ M ≤ 10^3). Следующие M строк
содержат по четыре целых числа числа: номера городов, которые соединяет
соответствующий маршрут, его пропускную способность и стоимость перевозки груза
по нему. Пропускные способности и стоимости не превосходят 10^5.
Гарантируется, что в графе поставок нет циклов отрицательной стоимости.

Формат вывода
В выходной файл выведите одно число — минимальную стоимость максимального объема
поставок из города 1 в город N.

*/

#include <algorithm>
#include <iostream>
#include <memory>
#include <set>
#include <vector>

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

class Graph {
 public:
  struct Edge;

  explicit Graph(size_t num_ver);

  void AddEdge(const Edge& edge);
  long long GetMinCostMaxFlow(size_t start, size_t end);

 private:
  const long long kInf = 1e18;

  vector<vector<shared_ptr<Edge>>> m_graph;
  vector<long long> m_dist, m_prev, m_price, m_flow;

  void ConuntPotential();
  void FindDistance(size_t start);
  bool UpdatePotentials(size_t start, size_t end);
};

struct Graph::Edge {
  size_t from, to;
  long long capacity, cost;
  shared_ptr<Edge> twin;

  Edge() = default;
  Edge(size_t from_n, size_t to_n, long long capacity_n, long long cost_n)
      : from(from_n), to(to_n), capacity(capacity_n), cost(cost_n) {}

  static void Connect(shared_ptr<Edge>& edge1, shared_ptr<Edge>& edge2);
};

void Graph::Edge::Connect(shared_ptr<Edge>& edge1, shared_ptr<Edge>& edge2) {
  edge1->twin = edge2;
  edge2->twin = edge1;
}

std::istream& operator>>(std::istream& inp, Graph::Edge& edge) {
  inp >> edge.from >> edge.to >> edge.capacity >> edge.cost;
  --edge.from;
  --edge.to;
  return inp;
}

Graph::Graph(size_t num_ver) : m_graph(num_ver) {}

void Graph::AddEdge(const Edge& edge) {
  auto edge1 = std::make_shared<Edge>(edge);
  auto edge2 = std::make_shared<Edge>(edge.to, edge.from, 0, -edge.cost);
  Edge::Connect(edge1, edge2);
  m_graph[edge1->from].push_back(edge1);
  m_graph[edge2->from].push_back(edge2);
}

void Graph::ConuntPotential() {
  m_price.assign(m_graph.size(), kInf);
  m_price[0] = 0;

  for (size_t ind = 0; ind < m_graph.size() - 1; ++ind) {
    for (size_t ver = 0; ver < m_graph.size(); ++ver) {
      for (auto& edge : m_graph[ver]) {
        if (edge->capacity > 0 &&
            m_price[edge->to] > m_price[ver] + edge->cost) {
          m_price[edge->to] = m_price[ver] + edge->cost;
        }
      }
    }
  }
}

void Graph::FindDistance(size_t start) {
  vector<bool> used(m_graph.size());
  set<pair<long long, size_t>> queue;
  m_dist[start] = 0;
  queue.insert({start, m_dist[start]});

  while (!queue.empty()) {
    size_t ind = queue.begin()->second;
    queue.erase(queue.begin());
    used[ind] = true;
    for (auto& edge : m_graph[ind]) {
      long long new_dist =
          m_dist[ind] + edge->cost + m_price[ind] - m_price[edge->to];
      if (!used[edge->to] && edge->capacity > 0 &&
          m_dist[edge->to] > new_dist) {
        queue.erase({m_dist[edge->to], edge->to});
        m_dist[edge->to] = new_dist;
        m_flow[edge->to] = std::min(m_flow[ind], edge->capacity);
        m_prev[edge->to] = ind;
        queue.insert({m_dist[edge->to], edge->to});
      }
    }
  }
}

bool Graph::UpdatePotentials(size_t start, size_t end) {
  m_prev.assign(m_graph.size(), -1);
  m_flow.assign(m_graph.size(), kInf);
  m_dist.assign(m_graph.size(), kInf);
  FindDistance(start);

  if (m_dist[end] >= kInf) {
    return false;
  }

  size_t ind = end;
  while (ind != 0) {
    for (auto& edge : m_graph[m_prev[ind]]) {
      if (edge->to == ind) {
        edge->capacity -= m_flow[end];
        edge->twin->capacity += m_flow[end];
      }
    }
    ind = m_prev[ind];
  }
  return true;
}

long long Graph::GetMinCostMaxFlow(size_t start, size_t end) {
  ConuntPotential();
  long long cost = 0;
  while (UpdatePotentials(start, end)) {
    for (size_t ind = 0; ind < m_graph.size(); ++ind) {
      m_price[ind] += m_dist[ind];
    }
    cost += m_flow[m_graph.size() - 1] * m_price[m_graph.size() - 1];
  }
  return cost;
}

signed main() {
  std::cin.tie(nullptr)->sync_with_stdio(false);
  std::cout.precision(20);

  size_t num_ver, num_edges;
  std::cin >> num_ver >> num_edges;

  Graph m_graph(num_ver);
  for (size_t ind = 0; ind < num_edges; ++ind) {
    Graph::Edge edge;
    std::cin >> edge;
    m_graph.AddEdge(edge);
  }

  std::cout << m_graph.GetMinCostMaxFlow(0, num_ver - 1) << '\n';
  std::cout.flush();

  return 0;
}
