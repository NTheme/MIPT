#include <algorithm>
#include <iostream>
#include <vector>

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

std::vector<long long> GetSeq(const std::vector<long long>& array) {
  std::vector<std::vector<long long>> bdp;

  bdp.push_back({});
  for (size_t index = 0; index < array.size(); ++index) {
    for (size_t pos = bdp.size(); pos > 0; --pos) {
      if (pos == 1 ||
          (bdp[pos - 1].back() - array[index]) * (1 - 2 * ((int)pos % 2)) < 0) {
        if (pos == bdp.size()) {
          bdp.push_back(bdp[pos - 1]);
          bdp[pos].push_back(array[index]);
        } else if ((bdp[pos].back() - array[index]) * (1 - 2 * ((int)pos % 2)) <
                   0) {
          bdp[pos] = bdp[pos - 1];
          bdp[pos].push_back(array[index]);
        }
      }
    }
  }

  return bdp.back();
}

int main() {
  std::cin.tie(nullptr)->sync_with_stdio(false);

  size_t n;
  std::cin >> n;
  std::vector<long long> array(n);
  std::cin >> array;

  auto ret1 = GetSeq(array);
  for (auto& p : array) {
    p = -p;
  }
  auto ret2 = GetSeq(array);

  if (ret1.size() > ret2.size()) {
    std::cout << ret1.size() << '\n' << ret1;
  } else {
    for (auto& p : ret2) {
      p = -p;
    }
    std::cout << ret2.size() << '\n' << ret2;
  }

  return 0;
}
