/*

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

*/

#include <algorithm>
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

struct Query {
  string type;
  pair<size_t, size_t> data;

  Query() = default;
};

std::istream& operator>>(std::istream& inp, Query& query) {
  inp >> query.type >> query.data.first >> query.data.second;
  return inp;
}

vector<string> ProcessQueries(size_t num_vertexes, const vector<Query>& queries) {
  DSU set(num_vertexes);
  vector<string> answers;
  for (auto it = queries.rbegin(); it != queries.rend(); ++it) {
    size_t ver1 = it->data.first - 1;
    size_t ver2 = it->data.second - 1;
    if (it->type == "ask") {
      answers.push_back(set.Find(ver1) == set.Find(ver2) ? "YES\n" : "NO\n");
    } else if (it->type == "cut") {
      set.Union(ver1, ver2);
    }
  }
  std::reverse(answers.begin(), answers.end());
  return answers;
}

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  size_t num_vertexes;
  size_t num_edges;
  size_t num_queries;
  std::cin >> num_vertexes >> num_edges >> num_queries;
  vector<pair<size_t, size_t>> edges(num_edges);
  vector<Query> queries(num_queries);
  std::cin >> edges >> queries;

  std::cout << ProcessQueries(num_vertexes, queries);
  std::cout.flush();
  return 0;
}
