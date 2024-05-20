/*

*/

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

const int kMod = 1e9 + 9;

struct Air {
  long long x, y;
  unsigned long long k;
  Air() : x(0), y(0), k(0) {}
  Air(long long xn, long long yn, unsigned long long kn)
      : x(xn), y(yn), k(kn) {}
};

std::istream& operator>>(std::istream& inp, Air& array) {
  inp >> array.x >> array.y >> array.k;
  return inp;
}

template <typename Type>
std::vector<std::vector<Type>> operator*(
    const std::vector<std::vector<Type>>& left,
    const std::vector<std::vector<Type>>& right) {
  std::vector<std::vector<Type>> result(left.size(),
                                        std::vector<Type>(right[0].size()));
  for (size_t line = 0; line < left.size(); ++line) {
    for (size_t column = 0; column < right[0].size(); ++column) {
      for (size_t index = 0; index < right.size(); ++index) {
        result[line][column] =
            (result[line][column] + left[line][index] * right[index][column]) %
            kMod;
      }
    }
  }
  return result;
}

template <typename Type>
std::vector<std::vector<Type>> Pow(const std::vector<std::vector<Type>>& matrix,
                                   unsigned long long pw) {
  std::vector<std::vector<Type>> m_pw = matrix;
  std::vector<std::vector<Type>> res(matrix.size(),
                                     std::vector<Type>(matrix.size()));
  for (size_t i = 0; i < res.size(); ++i) {
    res[i][i] = 1;
  }

  while (pw > 0) {
    if (pw % 2 == 1) {
      res = res * m_pw;
    }
    m_pw = m_pw * m_pw;
    pw /= 2;
  }

  return res;
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  int n, q;
  std::cin >> n >> q;
  std::vector<Air> air(n);
  std::cin >> air;
  std::vector<long long> line(q);
  std::cin >> line;

  std::vector<unsigned long long> ans(q);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < q; ++j) {
      long long b = air[i].y - air[i].x - line[j];
      if (b < 0) {
        continue;
      }

      std::vector<std::vector<unsigned long long>> fib = {{0, 1}};
      std::vector<std::vector<unsigned long long>> move = {{0, 1}, {1, 1}};
      fib = fib * Pow(move, b + 1);
      ans[j] = (ans[j] + (air[i].k % kMod) * fib[0][0]) % kMod;
    }
  }

  std::cout << ans;

  return 0;
}
