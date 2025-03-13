/*

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

class Graph {
 public:
  Graph(const vector<std::vector<size_t>>& matrix);
  vector<vector<size_t>> GetDistances();

 private:
  vector<vector<size_t>> graph_;
};

Graph::Graph(const std::vector<std::vector<size_t>>& matrix) : graph_(matrix) {}

vector<vector<size_t>> Graph::GetDistances() {
  vector<vector<size_t>> dist = graph_;
  for (size_t k = 0; k < graph_.size(); ++k) {
    for (size_t i = 0; i < graph_.size(); ++i) {
      for (size_t j = 0; j < graph_.size(); ++j) {
        dist[i][j] =
            std::max(dist[i][j], (size_t)(dist[i][k] + dist[k][j] == 2));
      }
    }
  }
  return dist;
}

void Task() {
  size_t num_vertexes;
  std::cin >> num_vertexes;
  vector<vector<size_t>> matrix(num_vertexes, vector<size_t>(num_vertexes));
  for (auto& row : matrix) {
    std::cin >> row;
  }

  auto ret = Graph(matrix).GetDistances();
  for (auto& elem : ret) {
    std::cout << elem << '\n';
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);

  Task();

  return 0;
}
