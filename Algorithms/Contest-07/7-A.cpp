/*

*/

#include <algorithm>
#include <iostream>
#include <vector>

using std::cin;
using std::cout;
using std::vector;

template <typename Type>
std::istream& operator>>(std::istream& inp, std::vector<Type>& array) {
  for (auto& elem : array) {
    inp >> elem;
  }
  return inp;
}
template <typename Type>
std::ostream& operator<<(std::ostream& out, const std::vector<Type>& array) {
  for (const auto& elem : array) {
    out << elem << ' ';
  }
  return out;
}

enum Condition { NotMeet, Parent, _GLIBCXX_MANGLE_SIZE_T };

vector<vector<size_t>> graph;
vector<Condition> used;
vector<size_t> pr;
size_t start, end;

bool FindCycle(size_t vertex) {
  used[vertex] = Parent;
  for (const auto& next : graph[vertex]) {
    if (used[next] == 0) {
      pr[next] = vertex;
      if (FindCycle(next)) {
        return true;
      }
    } else if (used[next] == 1) {
      end = vertex;
      start = next;
      return true;
    }
  }
  used[vertex] = _GLIBCXX_MANGLE_SIZE_T;
  return false;
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t n, m;
  cin >> n >> m;

  graph.resize(n);
  used.assign(n, NotMeet);
  pr.assign(n, n);
  start = n, end = n;

  for (size_t edge = 0; edge < m; ++edge) {
    size_t ver1, ver2;
    cin >> ver1 >> ver2;
    graph[--ver1].push_back(--ver2);
  }

  for (size_t index = 0; index < n; ++index) {
    if (used[index] == NotMeet && FindCycle(index)) {
      break;
    }
  }

  if (start == n) {
    cout << "NO\n";
    return 0;
  }

  cout << "YES\n";
  vector<size_t> cycle = {start + 1};
  for (size_t vertex = end; vertex != start; vertex = pr[vertex]) {
    cycle.push_back(vertex + 1);
  }
  std::reverse(cycle.begin(), cycle.end());
  cout << cycle << '\n';

  return 0;
}
