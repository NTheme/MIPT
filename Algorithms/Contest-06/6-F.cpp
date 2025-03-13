/*
B (1 балл, с ревью). Невозрастающая подпоследовательность
Ограничение времени	2 секунды
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt
Тут нет легенды, чистая классика.
Вам требуется написать программу, которая по заданной последовательности находит
максимальную невозрастающую её подпоследовательность (то есть такую
последовательность чисел a_i_1, a_i_2, … , a_i_k (i_1 < i_2 < … < i_k), что
a_i_1 ≥ a_i_2 ≥ … ≥ a_i_k и не существует последовательности с теми же
свойствами длиной k + 1).
Формат ввода
В первой строке задано число n — количество элементов последовательности (1 ≤ n
≤ 239017). В последующих строках идут сами числа последовательности a_i,
отделенные друг от друга произвольным количеством пробелов и переводов строки
(все числа не превосходят по модулю 2^31 − 2).
Формат вывода
Вам необходимо выдать в первой строке выходного файла число k — длину
максимальной невозрастающей подпоследовательности. В последующих строках должны
быть выведены (по одному числу в каждой строке) все номера элементов исходной
последовательности i_j, образующих искомую подпоследовательность. Номера
выводятся в порядке возрастания. Если оптимальных решений несколько, разрешается
выводить любое.
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

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  int n;
  std::cin >> n;
  std::vector<long long> array(n);
  std::cin >> array;
  array.insert(array.end(), array.begin(), array.end());
  int n_s = n;
  n *= 2;

  std::vector<std::vector<bool>> div(n, std::vector<bool>(2 * n));
  for (size_t i = 0; i < array.size(); ++i) {
    for (size_t j = 0; j < array.size(); ++j) {
      if (array[i] % array[j] == 0) {
        div[i][j] = true;
      }
    }
  }

  std::vector<std::vector<int>> dp(60, std::vector<int>(n, -1e9));
  std::vector<std::vector<int>> pr(60, std::vector<int>(n, -1));
  for (int i = 0; i < n; ++i) {
    dp[0][i] = i;
  }

  int iter = 0, pos = 0;
  for (size_t i = 1; i < dp.size(); ++i) {
    for (int j = i; j < n; ++j) {
      for (int k = j - 1; k >= std::max(0, j - n_s + 1); --k) {
        if (div[j][k] && j < n_s + dp[i - 1][k] && dp[i][j] < dp[i - 1][k]) {
          dp[i][j] = dp[i - 1][k];
          pr[i][j] = k;
          if (dp[i][j] > -1) {
            iter = i;
            pos = j;
          }
        }
      }
    }
  }

  std::cout << iter + 1 << '\n';
  std::vector<int> ans;
  for (; iter > -1; pos = pr[iter][pos], --iter) {
    ans.push_back(pos % n_s + 1);
  }

  std::reverse(ans.begin(), ans.end());
  std::cout << ans;

  return 0;

  return 0;
}