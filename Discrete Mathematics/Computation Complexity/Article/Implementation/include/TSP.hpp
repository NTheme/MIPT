/*--========================================--*\
* Author  : NTheme - All rights reserved
* Created : 1 December 2024, 4:16 AM
* File    : TSP.hpp
* Project : Salesman
\*--========================================--*/

#pragma once

#include <vector>

namespace nt {

std::vector<std::pair<size_t, size_t>> findMST(const std::vector<std::vector<size_t>> &weights);

class EulerGraph {
 public:
  explicit EulerGraph(size_t numVertices);

  void addEdge(size_t u, size_t v);
  std::vector<size_t> findEulerCycle();

  void clear();

 private:
  std::vector<std::pair<size_t, size_t>> m_edges;
  std::vector<std::vector<size_t>> m_graph;
  std::vector<bool> m_used;

  void buildCycle(size_t start, std::vector<size_t> &res);
  bool hasEdges(size_t v);
  size_t available(size_t v);
};

std::vector<size_t> TSPApproximation(const std::vector<std::vector<size_t>> &weights);

size_t getWeight(const std::vector<size_t> &cycle, const std::vector<std::vector<size_t>> &weights);

}  // namespace nt
