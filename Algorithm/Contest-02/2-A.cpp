/*
A (1 балл). Написание мешапа

Ограничение времени	2 секунды
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Известный мешапер создает гениальное произведение. У него есть фрагменты минуса
известной композиции, каждый фрагмент имеет вид отрезка с l_i по r_i секунды.
Ему хочется узнать, какие отрезки композиции у него есть, если их все
объединить.

Формат ввода
На первой строке идет число N (1 ≤ N ≤ 10^5) — число известных фрагментов
минуса. Далее идут N строк по два числа l_i, r_i (1 ≤ l_i ≤ r_i ≤ 10^9) — начало
и конец i-го фрагмента.

Формат вывода
Выведите число K — число известных отрезков минуса после объединения. Далее
выведите K строк по два числа l_i, r_i — границы i-го фрагмента. Выводить
фрагменты можно в любом порядке.
*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <deque>
#include <iostream>
#include <iterator>

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

template <class Iterator>
void Merge(Iterator first, Iterator last, Iterator mid) {
  std::deque<typename Iterator::value_type> temp;

  Iterator left = first, right = mid;
  while (left != mid and right != last) {
    if (*right < *left) {
      temp.push_back(*right++);
    } else {
      temp.push_back(*left++);
    }
  }
  temp.insert(temp.end(), left, mid);
  temp.insert(temp.end(), right, last);

  std::move(temp.begin(), temp.end(), first);
}

template <class Iterator>
void MergeSort(Iterator first, Iterator last) {
  if (last - first <= 1) {
    return;
  }

  Iterator mid = first + (last - first) / 2;
  MergeSort(first, mid);
  MergeSort(mid, last);
  Merge(first, last, mid);
}

void MakeEvents(std::deque<std::pair<int, int>>& seg,
                std::deque<std::pair<int, int>>& events) {
  for (auto& p : seg) {
    events.push_back({p.first, 0});
    events.push_back({p.second, 1});
  }
}

void UniteSegments(std::deque<std::pair<int, int>>& seg,
                   std::deque<std::pair<int, int>>& united) {
  std::deque<std::pair<int, int>> events;
  MakeEvents(seg, events);
  MergeSort(events.begin(), events.end());

  int left_open = 0, num_open = 0;
  for (auto& p : events) {
    if (p.second == 0) {
      if (num_open == 0) {
        left_open = p.first;
      }
      ++num_open;
    } else if (p.second == 1) {
      if (num_open == 1) {
        united.push_back({left_open, p.first});
      }
      --num_open;
    }
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(0), std::cout.tie(0);
  std::cout.precision(20);
  std::srand((unsigned int)time(NULL));

  size_t n = 0;
  std::cin >> n;

  std::deque<std::pair<int, int>> seg(n);
  std::cin >> seg;

  std::deque<std::pair<int, int>> united;
  UniteSegments(seg, united);

  std::cout << united.size() << '\n' << united;

  return 0;
}
