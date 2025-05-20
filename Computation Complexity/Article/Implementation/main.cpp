/*--========================================--*\
* Author  : NTheme - All rights reserved
* Created : 1 December 2024, 4:16 AM
* File    : main.cpp
* Project : Salesman
\*--========================================--*/

#include "TSP.hpp"

#include <iostream>
#include <vector>

template <typename Type>
std::istream &operator>>(std::istream &inp, std::vector<Type> &array) {
  for (auto &elem : array) {
    inp >> elem;
  }
  return inp;
}

template <typename Type>
std::ostream &operator<<(std::ostream &out, const std::vector<Type> &array) {
  for (const auto &elem : array) {
    out << elem << ' ';
  }
  return out;
}

int main() {
  size_t n;
  std::cout << "Enter number of vertices: ";
  std::cin >> n;

  std::vector<std::vector<size_t>> weights(n, std::vector<size_t>(n));
  std::cout << "Enter weights:" << std::endl;
  std::cin >> weights;

  std::vector<size_t> cycle = nt::TSPApproximation(weights);
  std::cout << "Cycle weight: " << nt::getWeight(cycle, weights) << "\nCycle: " << cycle << std::endl;

  return 0;
}
