/*
I (2 балла). Инверсии

Ограничение времени	1 секунда
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Напишите программу, которая для заданного массива A = ⟨a_1, a_2, ..., a_n⟩
находит количество пар (i, j) таких, что i < j и a_i > a_j. Обратите внимание на
то, что ответ может не влезать в int.

Формат ввода
Первая строка входного файла содержит натуральное число n (1 ≤ n ≤ 100000) —
количество элементов массива. Вторая строка содержит n попарно различных
элементов массива A — целых неотрицательных чисел, не превосходящих 10^9.

Формат вывода
В выходной файл выведите одно число — ответ на задачу.
*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <iostream>
#include <vector>

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
std::istream& operator>>(std::istream& inp, std::vector<T>& a) {
  for (auto& p : a) {
    inp >> p;
  }
  return inp;
}
template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& a) {
  for (auto& p : a) {
    out << p << '\n';
  }
  return out;
}

template <class Iterator>
long long Merge(Iterator first, Iterator last, Iterator mid) {
  std::vector<typename Iterator::value_type> temp;

  long long ans = 0;
  Iterator left = first, right = mid;
  while (left != mid and right != last) {
    if (*right < *left) {
      temp.push_back(*right++);
    } else {
      temp.push_back(*left++);
      ans += right - mid;
    }
  }

  ans += (right - mid) * (mid - left);
  temp.insert(temp.end(), left, mid);
  temp.insert(temp.end(), right, last);

  std::move(temp.begin(), temp.end(), first);

  return ans;
}

template <class Iterator>
long long MergeSort(Iterator first, Iterator last) {
  if (last - first <= 1) {
    return 0;
  }

  Iterator mid = first + (last - first) / 2;

  long long v1 = MergeSort(first, mid);
  long long v2 = MergeSort(mid, last);
  return v1 + v2 + Merge(first, last, mid);
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(0), std::cout.tie(0);
  std::cout.precision(20);
  std::srand((unsigned int)time(NULL));

  int n = 0;
  std::cin >> n;

  std::vector<int> seg(n);
  std::cin >> seg;

  std::cout << MergeSort(seg.begin(), seg.end());

  return 0;
}
