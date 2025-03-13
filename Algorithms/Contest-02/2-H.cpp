/*
H (1 балл, с ревью). Быстрая сортировка

Ограничение времени	1 секунда
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Нужно с помощью быстрой сортировки отсортировать массив.
Запрещено инклудить algorithm!

Формат ввода
В первой строке дано число N (1 ≤ N ≤ 105). Далее во втрой строке идет N чисел,
каждое с новой строки. Числа не превосходят по модулю 109.

Формат вывода
Вывести отсортированный по неубыванию массив.
*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <iostream>
#include <vector>

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

template <typename Iterator>
auto InternalDerandQuickSelect(Iterator first, Iterator last, int k,
                               Iterator buffer);

template <typename Iterator>
void BubbleSort(Iterator first, Iterator last) {
  if (std::distance(first, last) <= 1) {
    return;
  }

  bool changed = false;
  do {
    changed = false;
    for (Iterator p = first + 1; p < last; ++p) {
      if (*(p - 1) > *p) {
        std::swap(*(p - 1), *p);
        changed = true;
      }
    }
  } while (changed);
}

template <typename Iterator>
auto GetMedian(Iterator first, Iterator last, size_t len, Iterator buffer) {
  size_t middle = len / 2;

  Iterator pos = buffer;
  for (Iterator it = first; it + len <= last; it += len) {
    BubbleSort(it, it + len);
    *pos++ = *(it + middle);
  }
  return InternalDerandQuickSelect(buffer, pos,
                                   std::distance(first, last) / (2 * len), pos);
}

template <typename Iterator, typename Value>
void Partition(Iterator& posl, Iterator& posr, Value pivot) {
  for (Iterator p = posl; p <= posr; ++p) {
    if (*p < pivot) {
      std::swap(*p, *posl++);
    }
  }

  for (Iterator p = posr; p >= posl; --p) {
    if (*p > pivot) {
      std::swap(*p, *posr--);
    }
  }
}

template <typename Iterator>
auto InternalDerandQuickSelect(Iterator first, Iterator last, int k,
                               Iterator buffer) {
  if (std::distance(first, last) == 1) {
    return *first;
  }
  if (std::distance(first, last) < 5) {
    BubbleSort(first, last);
    return *(first + k - 1);
  }

  auto pivot = GetMedian(first, last, 5, buffer);

  Iterator posl = first;
  Iterator posr = last - 1;
  Partition(posl, posr, pivot);

  if (k <= std::distance(first, posl)) {
    return InternalDerandQuickSelect(first, posl, k, buffer);
  }
  if (k <= std::distance(first, posr + 1)) {
    return *posr;
  }
  return InternalDerandQuickSelect(posr + 1, last,
                                   k - std::distance(first, posr + 1), buffer);
}

template <typename Iterator>
void InternalDerandQuickSort(Iterator first, Iterator last, Iterator buffer) {
  if (std::distance(first, last) == 1) {
    return;
  }

  if (std::distance(first, last) < 5) {
    BubbleSort(first, last);
  } else {
    auto pivot = InternalDerandQuickSelect(
        first, last, std::distance(first, last) / 2, buffer);

    Iterator posl = first;
    Iterator posr = last - 1;
    Partition(posl, posr, pivot);
    InternalDerandQuickSort(first, posl, buffer);
    InternalDerandQuickSort(posr + 1, last, buffer);
  }
}

template <typename Iterator>
void DerandQuickSort(Iterator first, Iterator last) {
  size_t size = std::distance(first, last);
  std::vector<std::remove_reference_t<decltype(*first)>> buffer(size);
  std::vector<std::remove_reference_t<decltype(*first)>> array(size);

  size_t index = 0;
  for (Iterator p = first; p != last; ++p, ++index) {
    array[index] = *p;
  }

  InternalDerandQuickSort(array.begin(), array.end(), buffer.begin());

  index = 0;
  for (Iterator p = first; p != last; ++p, ++index) {
    *p = array[index];
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(0);
  std::cout.precision(20);
  std::srand((unsigned int)time(NULL));

  size_t n = 0;
  std::cin >> n;

  std::vector<int> numbers(n);
  std::cin >> numbers;
  DerandQuickSort(numbers.begin(), numbers.end());
  std::cout << numbers;

  return 0;
}
