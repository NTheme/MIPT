/*
D (3 балла, с ревью). Гоблины

Ограничение времени	1 секунда
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Гоблины Мглистых гор очень любях ходить к своим шаманам. Так как гоблинов много,
к шаманам часто образуются очень длинные очереди. А поскольку много гоблинов в
одном месте быстро образуют шумную толку, которая мешает шаманам проводить
сложные медицинские манипуляции, последние решили установить некоторые правила
касательно порядка в очереди.
Обычные гоблины при посещении шаманов должны вставать в конец очереди.
Привилегированные же гоблины, знающие особый пароль, встают ровно в ее середину,
причем при нечетной длине очереди они встают сразу за центром.
Так как гоблины также широко известны своим непочтительным отношением ко
всяческим правилам и законам, шаманы попросили вас написать программу, которая
бы отслеживала порядок гоблинов в очереди.

Формат ввода
В первой строке входных данный записано число N (1 ≤ N ≤ 10^5) — количество
запросов к программе. Следующие N строк содержат описание запросов в формате:
«+ i» — гоблин с номером i (1 ≤ i ≤ N) встает в конец очереди;
«* i» — привилегированный гоблин с номером i встает в середину очереди;
«-» — первый гоблин из очереди уходит к шаманам. Гарантируется, что на момент
такого запроса очередь не пуста.

Формат вывода
Для каждого запроса типа «-»
программа должна вывести номер гоблина, который должен зайти к шаманам.
*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <algorithm>
#include <deque>
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

void PushSimple(std::deque<int>& queue, std::pair<size_t, size_t>& right_seg,
                int number) {
  queue[right_seg.second++] = number;
}

void PushPriority(std::deque<int>& queue, std::pair<size_t, size_t>& left_seg,
                  std::pair<size_t, size_t>& right_seg, int number) {
  if ((left_seg.second - left_seg.first) ==
      (right_seg.second - right_seg.first) + 1) {
    queue[--right_seg.first] = number;
  } else {
    queue[left_seg.second++] = number;
  }
}

void BalanceHalfs(std::deque<int>& queue, std::pair<size_t, size_t>& left_seg,
                  std::pair<size_t, size_t>& right_seg) {
  if ((right_seg.second - right_seg.first) >
      (left_seg.second - left_seg.first)) {
    queue[left_seg.second++] = queue[right_seg.first];
    queue[right_seg.first++] = -1;
  }
}

std::deque<int> CountSequence(std::deque<std::pair<char, int>>& input) {
  std::deque<int> queue(2 * input.size(), -1);
  std::pair<size_t, size_t> left_seg = {0, 0},
                            right_seg = {input.size(), input.size()};

  for (auto& p : input) {
    if (p.first == '+') {
      PushSimple(queue, right_seg, p.second);
    } else if (p.first == '*') {
      PushPriority(queue, left_seg, right_seg, p.second);
    } else if (p.first == '-') {
      left_seg.first++;
    }

    BalanceHalfs(queue, left_seg, right_seg);
  }

  return std::deque<int>(queue.begin(), queue.begin() + left_seg.first);
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(0), std::cout.tie(0);
  std::cout.precision(20);
  std::srand((unsigned int)time(NULL));

  size_t n = 0;
  std::cin >> n;

  std::deque<std::pair<char, int>> input(n);
  for (auto& p : input) {
    std::cin >> p.first;
    if (p.first == '+' || p.first == '*') {
      std::cin >> p.second;
    }
  }

  std::cout << CountSequence(input);

  return 0;
}
