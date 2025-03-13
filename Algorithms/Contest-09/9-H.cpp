/*

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

*/

#include <algorithm>
#include <iostream>
#include <vector>

using std::pair;
using std::string;
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

vector<vector<long long>> graph;
vector<int> set_pairs;
vector<bool> used;

bool Kunn(long long ver) {
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

void DFS(long long ver) {
  used[ver] = true;

  for (const auto& to : graph[ver]) {
    if (!used[to]) {
      DFS(to);
    }
  }
}

void Task() {
  long long mmm, nnn, aaa, bbb;
  std::cin >> mmm >> nnn >> aaa >> bbb;
  long long num_ver = mmm * nnn;
  graph.resize(num_ver);
  used.assign(num_ver, false);
  set_pairs.assign(num_ver, -1);

  vector<string> field(mmm);
  std::cin >> field;

  long long num = 0;
  for (long long xind = 0; xind < mmm; ++xind) {
    for (long long yind = 0; yind < nnn; ++yind) {
      if (field[xind][yind] == '.') {
        ++num;
        continue;
      }
      if ((xind + yind) % 2 == 1) {
        continue;
      }

      if (xind > 0 && field[xind - 1][yind] == '*') {
        graph[xind * nnn + yind].push_back((xind - 1) * nnn + yind);
      }
      if (yind > 0 && field[xind][yind - 1] == '*') {
        graph[xind * nnn + yind].push_back(xind * nnn + yind - 1);
      }
      if (xind < mmm - 1 && field[xind + 1][yind] == '*') {
        graph[xind * nnn + yind].push_back((xind + 1) * nnn + yind);
      }
      if (yind < nnn - 1 && field[xind][yind + 1] == '*') {
        graph[xind * nnn + yind].push_back(xind * nnn + yind + 1);
      }
    }
  }

  for (long long ver = 0; ver < num_ver; ++ver) {
    used.assign(num_ver, false);
    Kunn(ver);
  }

  vector<pair<long long, long long>> pairs;
  for (long long ver = 0; ver < num_ver; ++ver) {
    if (set_pairs[ver] != -1) {
      pairs.emplace_back(set_pairs[ver] + 1, ver + 1);
    }
  }

  std::cout << std::min(
                   (mmm * nnn - num) * bbb,
                   (long long)pairs.size() * aaa +
                       (mmm * nnn - num - (long long)pairs.size() * 2) * bbb)
            << '\n';
}

signed main() {
  std::cin.tie(nullptr)->sync_with_stdio(false);
  std::cout.precision(20);

  Task();

  std::cout.flush();
  return 0;
}
