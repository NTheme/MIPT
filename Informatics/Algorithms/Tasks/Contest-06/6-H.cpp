/*
H (1 балл). Общий танец лоурайдеров

Ограничение времени	1 секунда
Ограничение памяти	64Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Цезарь вызвал вас на танцевальный баттл на заниженных машинах. Как вы может
помните, там надо проклеивать стрелочки в нужные моменты времени. В этот же раз
вам дана последовательность движений машин Сиджея и Цезаря. Надо найти их
наибольшую общую подпоследовательность или общий танец лоурайдеров.

Формат ввода
Первая и вторая строки входа содержат две непустые строки, каждая из которых
состоит из строчных латинских букв — коды движений в танце. Длина каждой строки
не превосходит 1000.

Формат вывода
В первой строке выведите целое число k — длину общего танца. Во второй выведите
k целых чисел — индексы символов в первой строке, отсортированные по
возрастанию. В третьей, аналогично — отсортированные по возрастанию индексы во
второй строке. Символы в последовательности занумерованы с 1.

Если способов выбрать общий танец несколько, выведите любой из них.
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

std::pair<std::vector<size_t>, std::vector<size_t>> LCS(
    const std::string& first_str, const std::string& second_str) {
  std::vector<std::vector<size_t>> lcs(
      first_str.size() + 1, std::vector<size_t>(second_str.size() + 1));
  std::vector<std::vector<std::pair<size_t, size_t>>> ancestor(
      first_str.size() + 1,
      std::vector<std::pair<size_t, size_t>>(second_str.size() + 1));
  std::vector<size_t> first_seq, second_seq;

  for (size_t first = 1; first <= first_str.size(); ++first) {
    for (size_t second = 1; second <= second_str.size(); ++second) {
      if (first_str[first - 1] == second_str[second - 1]) {
        lcs[first][second] = lcs[first - 1][second - 1] + 1;
        ancestor[first][second] = {first - 1, second - 1};
      } else if (lcs[first - 1][second] >= lcs[first][second - 1]) {
        lcs[first][second] = lcs[first - 1][second];
        ancestor[first][second] = {first - 1, second};
      } else {
        lcs[first][second] = lcs[first][second - 1];
        ancestor[first][second] = {first, second - 1};
      }
    }
  }

  size_t first = first_str.size(), second = second_str.size();
  while (first != 0 && second != 0) {
    auto[first_prev, second_prev] = ancestor[first][second];
    if (first_prev == first - 1 && second_prev == second - 1) {
      first_seq.push_back(first);
      second_seq.push_back(second);
      --first, --second;
    } else if (first_prev == first - 1 && second_prev == second) {
      --first;
    } else {
      --second;
    }
  }
  std::reverse(first_seq.begin(), first_seq.end());
  std::reverse(second_seq.begin(), second_seq.end());

  return {first_seq, second_seq};
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  std::string first_str, second_str;
  std::cin >> first_str >> second_str;

  auto[first, second] = LCS(first_str, second_str);
  std::cout << first.size() << '\n' << first << '\n' << second;

  return 0;
}
