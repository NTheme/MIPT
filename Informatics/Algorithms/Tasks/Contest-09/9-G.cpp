/*

*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <algorithm>
#include <iostream>
#include <vector>

using std::pair;
using std::vector;

template <typename TypeFirst, typename TypeSecond>
std::istream& operator>>(std::istream& inp, pair<TypeFirst, TypeSecond>& pair) {
  inp >> pair.first >> pair.second;
  return inp;
}
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
    out << elem << '\n';
  }
  return out;
}

vector<vector<size_t>> graph;
vector<int> set_pairs;
vector<bool> used;

bool Kunn(size_t ver) {
  if (used[ver]) {
    return false;
  }
  used[ver] = true;
  for (const auto& to : graph[ver]) {
    if (set_pairs[to] == -1 || Kunn(set_pairs[to])) {
      set_pairs[to] = ver;
      return true;
    }
  }
  return false;
}

void Task() {
  size_t num_vertexes1;
  size_t num_vertexes2;
  std::cin >> num_vertexes1 >> num_vertexes2;

  graph.resize(num_vertexes1);
  used.assign(num_vertexes1, false);
  set_pairs.assign(num_vertexes2, -1);

  for (size_t ver = 0; ver < num_vertexes1; ++ver) {
    size_t other;
    std::cin >> other;
    while (other != 0) {
      graph[ver].push_back(other - 1);
      std::cin >> other;
    }
  }

  for (size_t ver = 0; ver < num_vertexes1; ++ver) {
    used.assign(num_vertexes1, false);
    Kunn(ver);
  }

  vector<pair<size_t, size_t>> pairs;
  for (size_t ver = 0; ver < num_vertexes2; ++ver) {
    if (set_pairs[ver] != -1) {
      pairs.emplace_back(set_pairs[ver] + 1, ver + 1);
    }
  }

  std::cout << pairs.size() << '\n' << pairs << '\n';
}

signed main() {
  std::cin.tie(nullptr)->sync_with_stdio(false);
  std::cout.precision(20);

  Task();

  std::cout.flush();
  return 0;
}
