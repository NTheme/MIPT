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

size_t FindIndex(const std::vector<long long>& array,
                 const std::vector<size_t>& dp, size_t index) {
  size_t left = 0, right = dp.size();
  while (right - left > 1) {
    size_t middle = (left + right) >> 1;
    if (dp[middle] != array.size() && array[dp[middle]] >= array[index]) {
      left = middle;
    } else {
      right = middle;
    }
  }

  return right;
}

std::vector<size_t> MaxDecreasingSequence(const std::vector<long long>& array) {
  std::vector<size_t> dp(array.size() + 1, array.size());
  std::vector<size_t> anc(array.size(), array.size());

  for (size_t index = 0; index < array.size(); index++) {
    size_t ind = FindIndex(array, dp, index);
    anc[index] = dp[ind - 1];
    dp[ind] = index;
  }

  std::vector<size_t> sequence;
  for (size_t index = dp.size() - 1; index > 0; --index) {
    if (dp[index] != array.size()) {
      for (size_t way = dp[index]; way != array.size(); way = anc[way]) {
        sequence.push_back(way + 1);
      }
      break;
    }
  }
  std::reverse(sequence.begin(), sequence.end());
  return sequence;
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t n;
  std::cin >> n;
  std::vector<long long> array(n);
  std::cin >> array;

  auto sequence = MaxDecreasingSequence(array);
  std::cout << sequence.size() << '\n' << sequence;

  return 0;
}