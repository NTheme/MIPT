#include <algorithm>
#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>

using std::pair;
using std::set;
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

struct Vertex {
  std::string str;
  int heurist;

  Vertex(const std::string& strn, int heuristn);
  bool operator<(const Vertex& other) const;
};

Vertex::Vertex(const std::string& strn, int heuristn)
    : str(strn), heurist(heuristn) {}

bool Vertex::operator<(const Vertex& other) const {
  return heurist < other.heurist ||
         (heurist == other.heurist && str < other.str);
}

int GetHeurist2(const string& str) {
  const int kZeroPos = 16;
  int wrong = 0;
  for (int index = 0; index < kZeroPos; ++index) {
    int posx = 3;
    int posy = 3;
    if (str[index] != '0') {
      posx = (str[index] - '0' - 1) % 4;
      posy = (str[index] - '0' - 1) / 4;
    }
    wrong += std::abs(posx - index % 4 + posy - index / 4);
  }
  return wrong;
}

void GetNeighbour(const string& cur, vector<pair<char, string>>& neighbour) {
  const int kUBorder = 4;
  const int kDBorder = 12;

  int pos = cur.find('0');
  if (pos >= kUBorder) {
    neighbour.push_back({'D', cur});
    std::swap(neighbour.back().second[pos],
              neighbour.back().second[pos - kUBorder]);
  }
  if (pos < kDBorder) {
    neighbour.push_back({'U', cur});
    std::swap(neighbour.back().second[pos],
              neighbour.back().second[pos + kUBorder]);
  }
  if (pos % kUBorder < kUBorder - 1) {
    neighbour.push_back({'L', cur});
    std::swap(neighbour.back().second[pos], neighbour.back().second[pos + 1]);
  }
  if (pos % kUBorder > 0) {
    neighbour.push_back({'R', cur});
    std::swap(neighbour.back().second[pos], neighbour.back().second[pos - 1]);
  }
}

void AStar(const std::string& start, const std::string& end) {
  set<string> used;
  set<Vertex> queue;
  unordered_map<string, int> dist;
  unordered_map<string, pair<char, string>> prev;
  unordered_map<string, int> heurist;
  dist[start] = 0;
  prev[start] = {'0', "0"};
  heurist[start] = GetHeurist2(start);
  queue.insert(Vertex(start, heurist[start]));

  while (!queue.empty()) {
    Vertex cur = *(queue.begin());
    if (end == cur.str) {
      break;
    }
    queue.erase(queue.begin());
    used.insert(cur.str);

    vector<pair<char, string>> neighbour;
    GetNeighbour(cur.str, neighbour);
    for (const auto& to : neighbour) {
      int score = dist[cur.str] + 1;
      if (used.contains(to.second) && score >= dist[to.second]) {
        continue;
      }
      if (queue.contains(Vertex(to.second, heurist[to.second]))) {
        queue.erase(Vertex(to.second, heurist[to.second]));
      }
      prev[to.second] = {to.first, cur.str};
      dist[to.second] = score;
      heurist[to.second] = dist[to.second] + GetHeurist2(to.second);
      queue.insert(Vertex(to.second, heurist[to.second]));
    }
  }

  vector<char> path;
  for (pair<char, string> vertex = prev[end]; vertex.first != '0';
       vertex = prev[vertex.second]) {
    path.push_back(vertex.first);
  }
  std::reverse(path.begin(), path.end());
  std::cout << path.size() << '\n' << path << '\n';
}

void Task() {
  const int kSize = 16;

  string start = "0000000000000000";
  string end = "123456789:;<=>?0";
  for (int index = 0; index < kSize; ++index) {
    int num;
    std::cin >> num;
    start[index] = '0' + num;
  }

  int inv = 0;
  for (int index = 0; index < kSize; ++index) {
    if (start[index] != '0') {
      for (int pos = 0; pos < index; ++pos) {
        if (start[pos] > start[index]) {
          ++inv;
        }
      }
    }
  }

  for (int index = 0; index < 16; ++index) {
    if (start[index] == '0') {
      inv += 1 + index / 4;
    }
  }

  if (inv % 2 == 1) {
    std::cout << "-1\n";
  } else {
    AStar(start, end);
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);

  Task();

  return 0;
}