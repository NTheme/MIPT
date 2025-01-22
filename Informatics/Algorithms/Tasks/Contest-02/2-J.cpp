/*
J (2 балла). Меньший справа

Ограничение времени	1 секунда
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Вам дан массив a[] из N целых чисел. Надо для каждого элемента найти число
элементов справа строго меньших данного.

Формат ввода
На первой строке идет число N (1 ≤ N ≤ 105), на второй строке идет массив a[],
элементы которого не превосходят по модулю 109.

Формат вывода
Выведите массив b[], где b[i] равно числу элементов с большим, чем i, индексом и
значением строго меньшим, чем a[i].
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
    out << p << ' ';
  }
  return out;
}

template <class Iterator>
void Merge(Iterator first, Iterator last, Iterator mid,
           std::vector<int>::iterator inv_first) {
  std::vector<typename Iterator::value_type> temp_arr;

  Iterator left = first, right = mid;
  while (left != mid and right != last) {
    if (*right < *left) {
      temp_arr.push_back(*right++);
    } else {
      inv_first[left->second] += right - mid;
      temp_arr.push_back(*left++);
    }
  }

  while (left != mid) {
    inv_first[left->second] += right - mid;
    temp_arr.push_back(*left++);
  }
  temp_arr.insert(temp_arr.end(), right, last);

  std::move(temp_arr.begin(), temp_arr.end(), first);
}

template <class Iterator>
void MergeSort(Iterator first, Iterator last, std::vector<int>::iterator inv) {
  if (last - first <= 1) {
    return;
  }

  Iterator mid = first + (last - first) / 2;

  MergeSort(first, mid, inv);
  MergeSort(mid, last, inv);
  Merge(first, last, mid, inv);
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(0), std::cout.tie(0);
  std::cout.precision(20);
  std::srand((unsigned int)time(NULL));

  int n = 0;
  std::cin >> n;

  std::vector<std::pair<int, int>> arr(n);
  for (int i = 0; i < n; ++i) {
    std::cin >> arr[i].first;
    arr[i].second = i;
  }

  std::vector<int> inversion(n);
  MergeSort(arr.begin(), arr.end(), inversion.begin());
  std::cout << inversion;

  return 0;
}
