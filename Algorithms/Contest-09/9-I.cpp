/*

I (2 балла, с ревью). Минимальное контролирующее множество

Ограничение времени	0.5 секунд
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Требуется построить в двудольном графе минимальное контролирующее множество
(вершинное покрытие), если дано максимальное паросочетание. Формат ввода В
первой строке файла даны два числа m и n (1 ≤ m, n ≤ 4000) — размеры долей.
Каждая из следующих m строк содержит список ребер, выходящих из соответствующей
вершины первой доли. Этот список начинается с числа K_i (0 ≤ K_i ≤ n) —
количества ребер, после которого записаны вершины второй доли, соединенные с
данной вершиной первой доли, в произвольном порядке. Сумма всех K_i во входном
файле не превосходит 500000. Последняя строка файла содержит некоторое
максимальное паросочетание в этом графе — m чисел 0 ≤ L_i ≤ n — соответствующая
i-й вершине первой доли вершина второй доли, или 0, если i-я вершина первой доли
не входит в паросочетание.

Формат вывода
Первая строка содержит размер минимального контролирующего множества. Вторая
строка содержит количество вершин первой доли S, после которого записаны S чисел
— номера вершин первой доли, входящих в контролирующее множество, в возрастающем
порядке. Третья строка содержит описание вершин второй доли в аналогичном
формате.

*/

#include <algorithm>
#include <iostream>
#include <set>
#include <vector>

using std::pair;
using std::set;
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

void DFS(size_t ver, const vector<vector<size_t>>& graph, vector<bool>& used) {
  used[ver] = true;
  for (const auto& to : graph[ver]) {
    if (!used[to]) {
      DFS(to, graph, used);
    }
  }
}

class BipartiteGraph {
 public:
  BipartiteGraph(size_t left, size_t right, const vector<vector<size_t>>& graph,
                 const vector<size_t>& max_matching);

  pair<vector<size_t>, vector<size_t>> GetMVC() const;

 private:
  size_t m_left;
  size_t m_right;
  vector<pair<size_t, size_t>> m_edges;
  set<pair<size_t, size_t>> m_max_matching;
};

BipartiteGraph::BipartiteGraph(size_t left, size_t right,
                               const vector<vector<size_t>>& graph,
                               const vector<size_t>& max_matching)
    : m_left(left), m_right(right) {
  for (size_t index = 0; index < m_left; ++index) {
    for (const auto& ver : graph[index]) {
      m_edges.emplace_back(index, ver - 1 + m_left);
    }
  }
  for (size_t index = 0; index < m_left; ++index) {
    if (max_matching[index] != 0) {
      m_max_matching.emplace(index, max_matching[index] - 1 + m_left);
    }
  }
}

pair<vector<size_t>, vector<size_t>> BipartiteGraph::GetMVC() const {
  vector<vector<size_t>> graph(m_left + m_right);
  for (const auto& edge : m_edges) {
    if (m_max_matching.contains(edge)) {
      graph[edge.second].push_back(edge.first);
    } else {
      graph[edge.first].push_back(edge.second);
    }
  }

  set<size_t> free_ver;
  for (size_t index = 0; index < m_left; index++) {
    free_ver.insert(index);
  }
  for (const auto& row : graph) {
    for (size_t to : row) {
      free_ver.erase(to);
    }
  }

  vector<bool> used(m_left + m_right);
  for (size_t ver : free_ver) {
    DFS(ver, graph, used);
  }

  vector<size_t> left;
  vector<size_t> right;
  for (size_t index = 0; index < m_left; index++) {
    if (!used[index]) {
      left.push_back(index + 1);
    }
  }
  for (size_t index = 0; index < m_right; index++) {
    if (used[m_left + index]) {
      right.push_back(index + 1);
    }
  }
  return {left, right};
}

void Input(size_t& num_ver1, size_t& num_ver2, vector<vector<size_t>>& list,
           vector<size_t>& max_matching) {
  std::cin >> num_ver1 >> num_ver2;

  list.resize(num_ver1);
  for (size_t index = 0; index < num_ver1; ++index) {
    size_t size;
    std::cin >> size;
    list[index].resize(size);
    while (size-- > 0) {
      std::cin >> list[index][size];
    }
  }

  max_matching.resize(num_ver1);
  std::cin >> max_matching;
}

signed main() {
  std::cin.tie(nullptr)->sync_with_stdio(false);
  std::cout.precision(20);

  size_t num_ver1;
  size_t num_ver2;
  vector<vector<size_t>> list;
  vector<size_t> max_matching;
  Input(num_ver1, num_ver2, list, max_matching);

  BipartiteGraph graph(num_ver1, num_ver2, list, max_matching);
  auto mvc = graph.GetMVC();
  std::cout << mvc.first.size() + mvc.second.size() << '\n'
            << mvc.first.size() << ' ' << mvc.first << '\n'
            << mvc.second.size() << ' ' << mvc.second << '\n';
  std::cout.flush();
  return 0;
}
