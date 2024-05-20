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
  Forest();
  ~Forest();
  void AddString(const string& str, size_t index);
  vector<vector<size_t>> GetIncludings(const string& str);

 private:
  class Vertex;

  size_t num_words_;
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

Forest::Forest() : num_words_(0), root_(new Vertex()) {}

Forest::~Forest() { delete root_; }

void Forest::AddString(const string& str, size_t index) {
  Vertex* vertex = root_;
  for (const auto& symb : str) {
    if (vertex->m_to[symb] == nullptr) {
      vertex->m_to[symb] = new Vertex(vertex, symb);
    }
    vertex = vertex->m_to[symb];
  }

  ++num_words_;
  vertex->m_leaf = true;
  vertex->m_matches.push_back({index, str.size()});
}

vector<vector<size_t>> Forest::GetIncludings(const string& str) {
  Vertex* vertex = root_;
  vector<vector<size_t>> includings(num_words_);
  for (size_t index = 0; index < str.size(); ++index) {
    vertex = GetMove(vertex, str[index]);
    for (Vertex* cur = vertex; cur != root_; cur = GetLeafLink(cur)) {
      if (cur->m_leaf) {
        for (const auto& leaf : cur->m_matches) {
          includings[leaf.first].push_back(index + 2 - leaf.second);
        }
      }
    }
  }
  return includings;
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

  string text;
  size_t num_str;
  std::cin >> text >> num_str;

  Forest forest;
  for (size_t index = 0; index < num_str; ++index) {
    string str;
    std::cin >> str;
    forest.AddString(str, index);
  }

  auto includings = forest.GetIncludings(text);
  for (const auto& cur : includings) {
    std::cout << cur.size() << ' ' << cur;
  }

  std::cout.flush();
  return 0;
}
