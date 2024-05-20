#pragma once

#include <algorithm>
#include <iostream>
#include <queue>
#include <unordered_set>
#include <vector>

static inline const std::unordered_set<char> ALPHABET = {'a', 'b', 'c', '1'};

class Graph {
 public:
  struct Edge;

  Graph();

  size_t addVertexes(size_t count);
  void addEdge(size_t from, size_t to, char move);

  void setRoot(size_t vertex);
  void setTerminal(size_t vertex);

  size_t getSize() const;
  size_t getRoot() const;
  size_t getTerminal() const;
  const std::vector<Edge>& getEdges(size_t vertex) const;

 private:
  std::vector<std::vector<Edge>> m_graph;

  std::pair<bool, bool> m_set;
  std::pair<size_t, size_t> m_endings;
};

struct Graph::Edge {
  size_t to;
  char move;

  Edge(size_t to_m, char move_m);
};

Graph::Edge::Edge(size_t to_m, char move_m) : to(to_m), move(move_m) {}

Graph::Graph() : m_set(false, false) {}

size_t Graph::addVertexes(size_t count) {
  for (size_t index = 0; index < count; ++index) {
    m_graph.push_back({});
  }
  return getSize();
}

void Graph::addEdge(size_t from, size_t to, char move) { m_graph[from].emplace_back(to, move); }

void Graph::setRoot(size_t vertex) {
  m_endings.first = vertex;
  m_set.first = true;
}

void Graph::setTerminal(size_t vertex) {
  m_endings.second = vertex;
  m_set.second = true;
}

size_t Graph::getSize() const { return m_graph.size(); }

size_t Graph::getRoot() const { return m_endings.first; }

size_t Graph::getTerminal() const { return m_endings.second; }

const std::vector<Graph::Edge>& Graph::getEdges(size_t vertex) const { return m_graph[vertex]; }

Graph buildAutomaton(const std::string& str) {
  Graph graph;
  std::vector<std::pair<size_t, size_t>> stack;
  for (const auto& symb : str) {
    if (ALPHABET.contains(symb)) {
      auto size = graph.addVertexes(2);
      graph.addEdge(size - 2, size - 1, symb);
      stack.emplace_back(size - 2, size - 1);
    } else if (symb == '+') {
      auto back = stack.back();
      stack.pop_back();
      graph.addEdge(stack.back().first, back.first, '1');
      graph.addEdge(back.second, stack.back().second, '1');
    } else if (symb == '*') {
      graph.addEdge(stack.back().first, stack.back().second, '1');
      graph.addEdge(stack.back().second, stack.back().first, '1');
    } else if (symb == '.') {
      auto back = stack.back();
      stack.pop_back();
      graph.addEdge(stack.back().second, back.first, '1');
      stack.back().second = back.second;
    }
  }

  graph.setRoot(stack.front().first);
  graph.setTerminal(stack.front().second);
  return graph;
}

long long checkReaches(const std::string& regular, size_t k, size_t l) {
  auto graph = buildAutomaton(regular);

  std::queue<std::pair<size_t, size_t>> queue;
  std::vector<bool> used(graph.getSize() * k, false);
  std::vector<long long> terminal_reaches(k, -1);

  queue.emplace(graph.getRoot(), 0);
  while (!queue.empty()) {
    auto vertex = queue.front();
    queue.pop();

    if (used[vertex.first]) {
      continue;
    }
    used[vertex.first] = true;
    if (vertex.first / k == graph.getTerminal()) {
      terminal_reaches[vertex.second % k] = vertex.second;
    }

    for (const auto& edge : graph.getEdges(vertex.first / k)) {
      if (edge.move == '1') {
        queue.push({edge.to * k + vertex.second % k, vertex.second});
      } else {
        queue.push({edge.to * k + (vertex.second + 1) % k, vertex.second + 1});
      }
    }
  }

  return terminal_reaches[l];
}
