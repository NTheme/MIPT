/*

*/

#include <algorithm>
#include <cmath>
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

const size_t kMod = 1e9 + 7;

struct Border {
  unsigned long long xl, xr;
  size_t y;

  Border() : xl(0), xr(0), y(0) {}
};

std::istream& operator>>(std::istream& inp, Border& br) {
  inp >> br.xl >> br.xr >> br.y;
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

  const size_t kY = 17;

  size_t n = 0;
  unsigned long long k = 0;
  std::cin >> n >> k;

  std::vector<Border> array(n);
  std::cin >> array;
  array.back().xr = k;

  std::vector<std::vector<unsigned long long>> cl(
      kY, std::vector<unsigned long long>(1));
  cl[kY - 1][0] = 1;

  for (const auto& br : array) {
    std::vector<std::vector<unsigned long long>> move(
        kY, std::vector<unsigned long long>(kY));
    for (size_t i = 0; i <= br.y; ++i) {
      if (i > 0) {
        move[kY - i - 1][kY - i] = 1;
      }
      if (i < std::min(kY - 1, br.y)) {
        move[kY - i - 1][kY - i - 2] = 1;
      }
      move[kY - i - 1][kY - i - 1] = 1;
    }
    cl = Pow(move, br.xr - br.xl) * cl;
  }

  std::cout << cl[kY - 1] << '\n';

  return 0;
}
