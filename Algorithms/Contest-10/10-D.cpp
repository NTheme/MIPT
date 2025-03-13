/******************************************
 *  Author : NThemeDEV
 *  Created : Sat Sep 20 2023
 *  File : 10-C.cpp
 ******************************************/

/*
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")
*/

#include <algorithm>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

using std::pair;
using std::shared_ptr;
using std::string;
using std::unordered_map;
using std::vector;

template <typename TypeFirst, typename TypeSecond>
std::istream& operator>>(std::istream& inp, pair<TypeFirst, TypeSecond>& pair) {
  inp >> pair.first >> pair.second;
  return inp;
}
template <typename TypeFirst, typename TypeSecond>
std::ostream& operator<<(std::ostream& out,
                         const pair<TypeFirst, TypeSecond>& pair) {
  out << pair.first << ' ' << pair.second << '\n';
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
  out << '\n';
  return out;
}

class Forest {
 public:
  explicit Forest(const vector<string>& pattr, uint8_t move);
  ~Forest();
  void UpdateIncl(const vector<string>& table, vector<vector<size_t>>& incl,
                  uint8_t move);

 private:
  class Vertex;

  Vertex* root_;

  Vertex* GetLink(Vertex* vertex);
  Vertex* GetLeafLink(Vertex* vertex);
  Vertex* GetMove(Vertex* vertex, char edge);
};

class Forest::Vertex {
 public:
  unordered_map<char, Vertex*> m_to;
  unordered_map<char, Vertex*> m_move;
  Vertex* m_link;
  Vertex* m_leaf_link;
  Vertex* m_parent;
  char m_edge;
  bool m_leaf;
  vector<pair<size_t, size_t>> m_matches;

  Vertex(Vertex* parent = nullptr, char move = 0);
  ~Vertex();
};

Forest::Vertex::Vertex(Vertex* parent, char move)
    : m_link(nullptr),
      m_leaf_link(nullptr),
      m_parent(parent),
      m_edge(move),
      m_leaf(false) {}

Forest::Vertex::~Vertex() {
  for (auto& ptr : m_to) {
    delete ptr.second;
  }
}

Forest::Forest(const vector<string>& pattr, uint8_t move)
    : root_(new Vertex()) {
  size_t x_ind = 0;
  size_t y_ind = 0;
  size_t x_dif = move;
  size_t y_dif = (move + 1) % 2;
  size_t x_size = pattr.size();
  size_t y_size = pattr[0].size();

  for (; x_ind < x_size && y_ind < y_size; x_ind += x_dif, y_ind += y_dif) {
    Vertex* vertex = root_;
    size_t x_cur = x_ind;
    size_t y_cur = y_ind;

    for (; x_cur < x_size && y_cur < y_size; x_cur += y_dif, y_cur += x_dif) {
      char symb = pattr[x_cur][y_cur];
      if (vertex->m_to[symb] == nullptr) {
        vertex->m_to[symb] = new Vertex(vertex, symb);
      }
      vertex = vertex->m_to[symb];
    }

    vertex->m_leaf = true;
    vertex->m_matches.push_back({x_ind + y_ind, (move == 1) ? y_size : x_size});
  }
}

Forest::~Forest() { delete root_; }

void Forest::UpdateIncl(const vector<string>& table,
                        vector<vector<size_t>>& incl, uint8_t move) {
  size_t x_ind = 0;
  size_t y_ind = 0;
  size_t x_dif = move;
  size_t y_dif = (move + 1) % 2;
  size_t x_size = table.size();
  size_t y_size = table[0].size();

  for (; x_ind < x_size && y_ind < y_size; x_ind += x_dif, y_ind += y_dif) {
    Vertex* vertex = root_;
    size_t x_cur = x_ind;
    size_t y_cur = y_ind;

    for (; x_cur < x_size && y_cur < y_size; x_cur += y_dif, y_cur += x_dif) {
      vertex = GetMove(vertex, table[x_cur][y_cur]);
      for (Vertex* cur = vertex; cur != root_; cur = GetLeafLink(cur)) {
        if (!cur->m_leaf) {
          continue;
        }
        for (const auto& [ind, size] : cur->m_matches) {
          if (move == 1 && x_cur >= ind) {
            ++incl[x_cur - ind][y_cur + 1 - size];
          } else if (move == 0 && y_cur >= ind) {
            ++incl[x_cur + 1 - size][y_cur - ind];
          }
        }
      }
    }
  }
}

Forest::Vertex* Forest::GetLink(Vertex* vertex) {
  if (vertex->m_link == nullptr) {
    if (vertex->m_parent == root_) {
      vertex->m_link = vertex->m_parent;
    } else {
      vertex->m_link = GetMove(GetLink(vertex->m_parent), vertex->m_edge);
    }
  }
  return vertex->m_link;
}

Forest::Vertex* Forest::GetLeafLink(Vertex* vertex) {
  if (vertex->m_leaf_link == nullptr) {
    auto* link = GetLink(vertex);
    if (link == root_) {
      vertex->m_leaf_link = root_;
    } else if (link->m_leaf) {
      vertex->m_leaf_link = link;
    } else {
      vertex->m_leaf_link = GetLeafLink(link);
    }
  }
  return vertex->m_leaf_link;
}

Forest::Vertex* Forest::GetMove(Vertex* vertex, char edge) {
  if (vertex->m_move[edge] == nullptr) {
    if (vertex->m_to[edge] != nullptr) {
      vertex->m_move[edge] = vertex->m_to[edge];
    } else if (vertex == root_) {
      vertex->m_move[edge] = root_;
    } else {
      vertex->m_move[edge] = GetMove(GetLink(vertex), edge);
    }
  }
  return vertex->m_move[edge];
}

size_t GetPositions(const vector<string>& table, const vector<string>& pattr) {
  Forest forest_row(pattr, 1);
  Forest forest_col(pattr, 0);
  vector<vector<size_t>> positions(table.size(),
                                   vector<size_t>(table[0].size()));

  forest_row.UpdateIncl(table, positions, 1);
  forest_col.UpdateIncl(table, positions, 0);

  size_t res = 0;
  for (size_t x_ind = 0; x_ind + pattr.size() <= table.size(); ++x_ind) {
    for (size_t y_ind = 0; y_ind + pattr[0].size() <= table[0].size();
         ++y_ind) {
      if (positions[x_ind][y_ind] + 2 >= pattr.size() + pattr[0].size()) {
        ++res;
      }
    }
  }
  return res;
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);

  size_t n_size;
  size_t m_size;
  std::cin >> n_size >> m_size;
  vector<string> table(n_size);
  std::cin >> table;

  size_t a_size;
  size_t b_size;
  std::cin >> a_size >> b_size;
  vector<string> pattr(a_size);
  std::cin >> pattr;

  std::cout << GetPositions(table, pattr) << '\n';

  std::cout.flush();
  return 0;
}
