/******************************************
 *  Author : NThemeDEV
 *  Created : Mon Oct 16 2023
 *  File : 11-О.cpp
 ******************************************/

/*
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")
*/

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

using std::pair;
using std::shared_ptr;
using std::string;
using std::vector;

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
struct Node {
  const size_t kAlphabet = 26;

  Node* suflink;
  Node* prev;

  size_t len;
  size_t cnt;
  size_t used;

  vector<Node*> go;

  Node();
  Node(const vector<Node*>& other);
};

Node::Node()
    : suflink(nullptr),
      prev(nullptr),
      len(0),
      cnt(0),
      used(0),
      go(kAlphabet, nullptr) {}

Node::Node(const vector<Node*>& other)
    : suflink(nullptr), prev(nullptr), len(0), cnt(0), used(0), go(other) {}

vector<Node*> all_nodes;

Node* Add(Node* last, size_t symb) {
  Node* new_node = new Node();
  all_nodes.push_back(new_node);
  new_node->prev = last;
  new_node->len = last->len + 1;

  while (last != nullptr && last->go[symb] == nullptr) {
    new_node->suflink = last;
    last->go[symb] = new_node;
    last = last->suflink;
  }

  if (last == nullptr) {
    return new_node;
  }

  if (last->go[symb]->prev == last) {
    new_node->suflink = last->go[symb];
  } else {
    Node* move = last->go[symb];
    Node* new_link = new Node(move->go);
    all_nodes.push_back(new_link);
    new_link->prev = last;
    new_link->len = last->len + 1;
    while (last != nullptr && last->go[symb] == move) {
      last->go[symb] = new_link;
      last = last->suflink;
    }
    new_link->suflink = move->suflink;
    move->suflink = new_link;
    new_node->suflink = new_link;
  }

  return new_node->prev->go[symb];
}

void Subautomation(Node* new_node, size_t time) {
  if (new_node == nullptr || new_node->used == time) {
    return;
  }
  new_node->used = time;
  new_node->cnt++;

  Subautomation(new_node->prev, time);
  Subautomation(new_node->suflink, time);
}

vector<size_t> ans;

void DFS(Node* new_node, size_t time) {
  new_node->used = time;
  ans[new_node->cnt] = std::max(ans[new_node->cnt], new_node->len);
  for (size_t index = 0; index < new_node->go.size(); ++index) {
    if (new_node->go[index] != nullptr && new_node->go[index]->used != time) {
      DFS(new_node->go[index], time);
    }
  }
}

signed main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(0);

  size_t nn;
  std::cin >> nn;
  ans.resize(nn + 1);

  size_t used = 0;
  vector<Node*> lst;
  Node* root = new Node();
  all_nodes.push_back(root);

  for (size_t index = 0; index < nn; ++index) {
    string buf;
    std::cin >> buf;
    Node* last = root;
    for (auto symb : buf) {
      last = Add(last, symb - 'a');
    }
    lst.push_back(last);
  }
  for (auto* new_node : lst) {
    ++used;
    Subautomation(new_node, used);
  }
  ++used;

  DFS(root, used);

  vector<size_t> ans1(nn + 1);
  ans1[nn] = ans[nn];

  for (size_t index = nn - 1; index > 1; --index) {
    ans1[index] = std::max(ans1[index + 1], ans[index]);
  }
  for (size_t index = 2; index <= nn; ++index) {
    std::cout << ans1[index] << '\n';
  }

  for (auto& ptr : all_nodes) {
    delete ptr;
  }
  std::cout.flush();
  return 0;
}
