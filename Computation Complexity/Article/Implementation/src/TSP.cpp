/*--========================================--*\
* Author  : NTheme - All rights reserved
* Created : 1 December 2024, 4:16 AM
* File    : TSP.cpp
* Project : Salesman
\*--========================================--*/

#include "TSP.hpp"
#include "PerfectMatching.h"

#include <set>
#include <iostream>
#include <algorithm>

namespace nt {

namespace {

std::vector<size_t> getOddVertices(size_t n, const std::vector<std::pair<size_t, size_t>> &edges) {
  std::vector<size_t> degrees(n);
  for (auto [u, v] : edges) {
    ++degrees[u];
    ++degrees[v];
  }

  std::vector<size_t> res;
  for (size_t v = 0; v < n; ++v) {
    if (degrees[v] % 2 == 1)
      res.push_back(v);
  }
  return res;
}

std::vector<size_t> dropDuplicates(const std::vector<size_t> &seq) {
  std::set<size_t> used;
  std::vector<size_t> res;
  for (size_t x : seq) {
    if (used.insert(x).second)
      res.push_back(x);
  }
  return res;
}

}  // namespace

EulerGraph::EulerGraph(size_t numVertices) : m_edges{}, m_graph(numVertices), m_used{} {
}

void EulerGraph::addEdge(size_t u, size_t v) {
  size_t e = m_edges.size();
  m_edges.emplace_back(u, v);
  m_used.push_back(false);
  m_graph[u].push_back(e);
  m_graph[v].push_back(e);
}

void EulerGraph::clear() {
  for (auto &neighbours : m_graph) {
    neighbours.clear();
  }
  m_edges.clear();
  m_used.clear();
}

std::vector<size_t> EulerGraph::findEulerCycle() {
  if (std::any_of(m_graph.begin(), m_graph.end(), [](const auto &neighbours) {
        return neighbours.size() % 2 == 1;
      })) {
    throw std::invalid_argument("No euler cycle!");
  }

  std::vector<size_t> cycle;
  buildCycle(0, cycle);
  clear();

  std::vector<bool> visited(m_graph.size());
  for (size_t v : cycle) {
    visited[v] = true;
  }
  if (!std::all_of(visited.begin(), visited.end(), [](bool b) {
        return b;
      })) {
    throw std::invalid_argument("No euler cycle!");
  }

  return cycle;
}

void EulerGraph::buildCycle(size_t start, std::vector<size_t> &res) {
  std::vector<size_t> cycle;
  for (size_t v = start; hasEdges(v); v = available(v)) {
    cycle.push_back(v);
  }

  for (size_t v : cycle) {
    if (hasEdges(v))
      buildCycle(v, res);
    res.push_back(v);
  }
}

bool EulerGraph::hasEdges(size_t v) {
  while (!m_graph[v].empty() && m_used[m_graph[v].back()]) {
    m_graph[v].pop_back();
  }
  return !m_graph[v].empty();
}

size_t EulerGraph::available(size_t v) {
  size_t e = m_graph[v].back();
  m_used[e] = true;
  return m_edges[e].first == v ? m_edges[e].second : m_edges[e].first;
}

std::vector<std::pair<size_t, size_t>> findMST(const std::vector<std::vector<size_t>> &weights) {
  size_t n = weights.size();
  std::vector<size_t> minEdge = weights[0];
  std::vector<size_t> edgeStart(n, 0);
  std::vector<bool> selected(n);
  selected[0] = true;

  std::vector<std::pair<size_t, size_t>> edges;
  for (size_t i = 1; i < n; ++i) {
    size_t u = n;
    for (size_t v = 0; v < n; ++v) {
      if (selected[v])
        continue;
      if (u == n || (minEdge[u] > minEdge[v])) {
        u = v;
      }
    }

    selected[u] = true;
    edges.emplace_back(u, edgeStart[u]);
    for (size_t v = 0; v < n; ++v) {
      if (selected[v])
        continue;
      if (weights[u][v] < minEdge[v]) {
        minEdge[v] = weights[u][v];
        edgeStart[v] = u;
      }
    }
  }

  return edges;
}

std::vector<size_t> TSPApproximation(const std::vector<std::vector<size_t>> &weights) {
  size_t n = weights.size();
  EulerGraph graph(n);

  std::vector<std::pair<size_t, size_t>> mstEdges = findMST(weights);
  for (auto [u, v] : mstEdges) {
    graph.addEdge(u, v);
  }

  std::vector<size_t> oddVertices = getOddVertices(n, mstEdges);
  size_t m = oddVertices.size();
  PerfectMatching matching((int)m, (int)(m * (m - 1)) / 2);
  for (size_t i = 0; i < m; ++i) {
    size_t u = oddVertices[i];
    for (size_t j = i + 1; j < m; ++j) {
      size_t v = oddVertices[j];
      std::ignore = matching.AddEdge((int)i, (int)j, (int)weights[u][v]);
    }
  }

  matching.options.verbose = false;
  matching.Solve();
  for (size_t i = 0; i < m; ++i) {
    size_t j = matching.GetMatch((int)i);
    if (i < j)
      graph.addEdge(oddVertices[i], oddVertices[j]);
  }

  return dropDuplicates(std::move(graph).findEulerCycle());
}

size_t getWeight(const std::vector<size_t> &cycle, const std::vector<std::vector<size_t>> &weights) {
  size_t weight = 0;
  for (size_t i = 0; i < cycle.size(); ++i) {
    size_t j = i + 1;
    if (j == cycle.size())
      j = 0;
    weight += weights[cycle[i]][cycle[j]];
  }
  return weight;
}

}  // namespace nt
