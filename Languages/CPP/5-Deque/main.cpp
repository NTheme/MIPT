/*

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")
*/

#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>

#include "deque.h"

using std::pair;
using std::vector;

template <typename TypeLeft, typename TypeRight>
std::istream& operator>>(std::istream& inp, pair<TypeLeft, TypeRight>& pair) {
  inp >> pair.first >> pair.second;
  return inp;
}
template <typename TypeLeft, typename TypeRight>
std::ostream& operator<<(std::ostream& out, const pair<TypeLeft, TypeRight>& pair) {
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

int main() {
  std::cin.tie(nullptr)->sync_with_stdio(false);
  std::cout.precision(20);

  Deque<int>::iterator a;
  Deque<int> b(5);
  Deque<int>::const_iterator c = b.cbegin();

  return 0;
}
