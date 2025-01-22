/*

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

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
  DSU(size_t size);

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

struct Edge {
  size_t ver1 = 0;
  size_t ver2 = 0;
  size_t weight = 0;

  Edge() = default;
  Edge(size_t from_n, size_t to_n, size_t weight_n) : ver1(from_n), ver2(to_n), weight(weight_n) {}

  std::strong_ordering operator<=>(const Edge& other) const;
};

std::strong_ordering Edge::operator<=>(const Edge& other) const {
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

std::istream& operator>>(std::istream& inp, Edge& edge) {
  inp >> edge.weight;
  return inp;
}

size_t FindSpanTreeWeight(size_t num_vertexes, vector<Edge> edges) {
  DSU set(num_vertexes);
  sort(edges.begin(), edges.end());

  size_t cost = 0;
  for (const auto& edge : edges) {
    if (set.Find(edge.ver1) != set.Find(edge.ver2)) {
      cost += edge.weight;
      set.Union(edge.ver1, edge.ver2);
    }
  }
  return cost;
}

void Task() {
  size_t num_vertexes;
  std::cin >> num_vertexes;
  vector<Edge> edges;
  for (size_t ver1 = 0; ver1 < num_vertexes; ++ver1) {
    for (size_t ver2 = 0; ver2 < num_vertexes; ++ver2) {
      size_t weight;
      std::cin >> weight;
      if (ver1 != ver2) {
        edges.emplace_back(ver1, ver2, weight);
      }
    }
  }
  for (size_t index = 0; index < num_vertexes; ++index) {
    size_t weight;
    std::cin >> weight;
    edges.emplace_back(index, num_vertexes, weight);
  }

  std::cout << FindSpanTreeWeight(num_vertexes + 1, edges);
}

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  Task();

  std::cout.flush();
  return 0;
}
