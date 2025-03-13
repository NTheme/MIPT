/*

*/

#include <algorithm>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>

using std::pair;
using std::queue;
using std::string;
using std::unordered_map;
using std::vector;

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
    out << elem;
  }
  return out;
}

void Fill(unordered_map<string, size_t>& perms,
          vector<vector<pair<size_t, char>>>& graph) {
  for (string cur = "012345678"; !perms.contains(cur);
       std::next_permutation(cur.begin(), cur.end())) {
    perms[cur] = perms.size();
  }

  const size_t kShift = 3;
  const size_t kShut = 6;

  graph.resize(perms.size());
  for (const auto& perm : perms) {
    size_t pos = perm.first.find('0');
    if (pos > 2) {
      string nperm = perm.first;
      std::swap(nperm[pos], nperm[pos - kShift]);
      graph[perm.second].push_back({perms[nperm], 'U'});
    }
    if (pos < kShut) {
      string nperm = perm.first;
      std::swap(nperm[pos], nperm[pos + kShift]);
      graph[perm.second].push_back({perms[nperm], 'D'});
    }
    if (pos % kShift < 2) {
      string nperm = perm.first;
      std::swap(nperm[pos], nperm[pos + 1]);
      graph[perm.second].push_back({perms[nperm], 'R'});
    }
    if (pos % kShift > 0) {
      string nperm = perm.first;
      std::swap(nperm[pos], nperm[pos - 1]);
      graph[perm.second].push_back({perms[nperm], 'L'});
    }
  }
}

void Task() {
  unordered_map<string, size_t> perms;
  vector<vector<pair<size_t, char>>> graph;
  Fill(perms, graph);

  const size_t kShop = 9;

  string str = "000000000";
  for (size_t index = 0; index < kShop; ++index) {
    std::cin >> str[index];
  }
  size_t start = perms[str];
  size_t end = perms["123456780"];

  queue<int> queue;
  queue.push(start);
  vector<bool> used(graph.size());
  vector<size_t> dist(graph.size());
  vector<pair<size_t, char>> prev(graph.size());
  used[start] = true;
  prev[start] = {graph.size(), 'O'};
  while (!queue.empty()) {
    size_t vertex = queue.front();
    queue.pop();
    for (const auto& to : graph[vertex]) {
      if (!used[to.first]) {
        used[to.first] = true;
        queue.push(to.first);
        dist[to.first] = dist[vertex] + 1;
        prev[to.first] = {vertex, to.second};
      }
    }
  }

  if (!used[end]) {
    std::cout << "-1\n";
  } else {
    vector<char> path;
    for (pair<size_t, char> vertex = prev[end]; vertex.first != graph.size();
         vertex = prev[vertex.first]) {
      path.push_back(vertex.second);
    }
    std::reverse(path.begin(), path.end());
    std::cout << path.size() << '\n' << path << '\n';
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);

  Task();

  return 0;
}
