/******************************************
 *  Author : NThemeDEV
 *  Created : Sat Sep 29 2023
 *  File : 10-I.cpp
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

static constexpr char NEW_LINE = '\n';

template <typename TypeFirst, typename TypeSecond>
std::istream& operator>>(std::istream& inp, pair<TypeFirst, TypeSecond>& pair) {
  inp >> pair.first >> pair.second;
  return inp;
}
template <typename TypeFirst, typename TypeSecond>
std::ostream& operator<<(std::ostream& out, const pair<TypeFirst, TypeSecond>& pair) {
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
  Forest();
  ~Forest();
  void AddString(const string& str);
  bool CheckAcyclic();

 private:
  class Vertex;

  size_t size_;
  Vertex* root_;

  Vertex* GetLink(Vertex* vertex);
  Vertex* GetLeafLink(Vertex* vertex);
  Vertex* GetMove(Vertex* vertex, char edge);
  bool DFS(Vertex* vertex, vector<uint8_t>& used);
};

class Forest::Vertex {
 public:
  size_t m_index;
  unordered_map<char, Vertex*> m_to;
  unordered_map<char, Vertex*> m_move;
  Vertex* m_link;
  Vertex* m_leaf_link;
  Vertex* m_parent;
  char m_edge;
  bool m_leaf;

  Vertex(size_t index, Vertex* parent = nullptr, char move = 0);
  ~Vertex();
};

Forest::Vertex::Vertex(size_t index, Vertex* parent, char move)
    : m_index(index), m_link(nullptr), m_leaf_link(nullptr), m_parent(parent), m_edge(move), m_leaf(false) {}

Forest::Vertex::~Vertex() {
  for (auto& ptr : m_to) {
    delete ptr.second;
  }
}

Forest::Forest() : size_(1), root_(new Vertex(0)) {}

Forest::~Forest() { delete root_; }

void Forest::AddString(const string& str) {
  Vertex* vertex = root_;
  for (const auto& symb : str) {
    if (vertex->m_to[symb] == nullptr) {
      vertex->m_to[symb] = new Vertex(size_++, vertex, symb);
    }
    vertex = vertex->m_to[symb];
  }

  vertex->m_leaf = true;
}

bool Forest::CheckAcyclic() {
  vector<uint8_t> used(size_);

  return DFS(root_, used);
}

bool Forest::DFS(Vertex* vertex, vector<uint8_t>& used) {
  used[vertex->m_index] = 1;
  for (char symb = '0'; symb != '2'; ++symb) {
    auto* to = GetMove(vertex, symb);
    bool enter = true;
    for (Vertex* cur = to; cur != root_; cur = GetLeafLink(cur)) {
      if (cur->m_leaf) {
        enter = false;
        break;
      }
    }
    if (enter && ((used[to->m_index] == 0 && DFS(to, used)) || used[to->m_index] == 1)) {
      return true;
    }
  }
  used[vertex->m_index] = 2;
  return false;
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

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);

  size_t num_str;
  std::cin >> num_str;

  Forest forest;
  for (size_t index = 0; index < num_str; ++index) {
    string str;
    std::cin >> str;
    forest.AddString(str);
  }

  std::cout << (forest.CheckAcyclic() ? "TAK" : "NIE");

  std::cout.flush();
  return 0;
}
