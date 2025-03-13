/*
H (2 балла). Очередная суматоха в Отделе Тайн

Все языки	Golang 1.16
Ограничение времени	3 секунды	5 секунд
Ограничение памяти	256Mb	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Отдел Тайн Министерства магии состоит из двух стеллажей A и B. Известно, что в
стеллаже A пророчества на каждой полке отсортированы по порядку времени, когда
они были созданы. А в стеллаже B пророчества на каждой полке отсортированы с
точностью до наоборот. В стеллаже A n полок, а в B — m полок. При этом в обоих
стеллажах все полки вмещают по l пророчеств. Тому, кого нельзя называть,
известно, что все пророчества в отделе тайн хранятся в виде двух частей: одна в
стеллаже A, другая — в B. Также он знает, что половинки пророчеств соединяются
следующим образом: на полках Ai и Bj половинки пророчества соответствуют друг
другу, если найти такое k, что max(Aik, Bjk) минимален. Тогда такие Aik, Bjk и
будут нужными пророчествами. Если таких k несколько, то подойдет любое. Так как
Тому Реддлу осталось недолго, он просит вас искать такие k на его запросы вида
(i, j).

Формат ввода
На первой строке числа n, m, l (1 ≤ n, m ≤ 900; 1 ≤ l ≤ 3000). Следующие n строк
содержат описания полок Ai. Каждая полка описывается перечислением l элементов.
Время записи пророчеств — целые числа от 0 до 10^5 − 1. Далее число m и описание
массивов Bj в таком же формате. Полки и элементы внутри массива нумеруются с 1.
На следующей строке число запросов Тома Реддла q (1 ≤ q ≤ n ⋅ m). Следующие q
строк содержат пары чисел i, j (1 ≤ i ≤ n, 1 ≤ j ≤ m).

Формат вывода
Выведите q чисел от 1 до l – ответы на запросы.
*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <algorithm>
#include <deque>
#include <fstream>
#include <iostream>

template <typename T1, typename T2>
std::istream& operator>>(std::istream& inp, std::pair<T1, T2>& a) {
  inp >> a.first >> a.second;
  return inp;
}
template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& out, const std::pair<T1, T2>& a) {
  out << a.first << ' ' << a.second;
  return out;
}

template <typename T>
std::istream& operator>>(std::istream& inp, std::deque<T>& a) {
  for (auto& p : a) {
    inp >> p;
  }
  return inp;
}
template <typename T>
std::ostream& operator<<(std::ostream& out, const std::deque<T>& a) {
  for (auto& p : a) {
    out << p << '\n';
  }
  return out;
}

size_t FindHunch(std::deque<std::deque<int>>& a, std::deque<std::deque<int>>& b,
                 int len, int i, int j) {
  size_t l = 0, r = len - 1;
  while (r - l > 1) {
    size_t m = (l + r) / 2;

    if (a[i][m] < b[j][m]) {
      l = m;
    } else {
      r = m;
    }
  }

  return (std::max(a[i][l], b[j][l]) > std::max(a[i][r], b[j][r])) ? r + 1
                                                                   : l + 1;
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(0), std::cout.tie(0);
  std::cout.precision(20);
  std::srand((unsigned int)time(NULL));

  int n = 0, m = 0, l = 0;
  std::cin >> n >> m >> l;

  std::deque<std::deque<int>> a(n, std::deque<int>(l));
  for (auto& p : a) {
    std::cin >> p;
  }

  std::deque<std::deque<int>> b(m, std::deque<int>(l));
  for (auto& p : b) {
    std::cin >> p;
  }

  int q = 0;
  std::cin >> q;

  while (q-- > 0) {
    int i = 0, j = 0;
    std::cin >> i >> j;
    std::cout << FindHunch(a, b, l, i - 1, j - 1) << '\n';
  }

  return 0;
}
