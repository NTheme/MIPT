#include <iostream>
#include <type_traits>
#include "smart_pointers.h"

struct Node;

struct Next {
  SharedPtr<Node> shared;
  WeakPtr<Node> weak;
  Next(const SharedPtr<Node>& shared) : shared(shared) {}
  Next(const WeakPtr<Node>& weak) : weak(weak) {}
};

struct Node {
  static int constructed;
  static int destructed;

  int value;
  Next next;
  Node(int value) : value(value), next(SharedPtr<Node>()) { ++constructed; }
  Node(int value, const SharedPtr<Node>& next) : value(value), next(next) { ++constructed; }
  Node(int value, const WeakPtr<Node>& next) : value(value), next(next) { ++constructed; }
  ~Node() { ++destructed; }
};

int main() {
  std::cout << std::is_convertible_v<Node, Node>;
  std::cout.flush();
  return 0;
}