/*
B (1 балл, с ревью). k-я порядковая статистика

Ограничение времени	0.5 секунд
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

В этой задаче вам необходимо найти k-ую порядковую статистику (k-ое по
неубыванию) числовой последовательности A, элементы которой задаются следующим
образом: Ai = (A(i - 1) * 123 + A(i - 2) * 45) % (10^7 + 4321)

Формат ввода
Вам даны n, k, A0, A1 (1 ≤ k ≤ n ≤ 10^7, 0 ≤ Ai < 10^7 + 4321) — количество
чисел в числовой последовательности, k из условия и первые два числа числовой
последовательности.

Формат вывода
Выведите k-ую порядковую статистику.
*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <iostream>
#include <vector>

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
auto DerandQuickSelect(Iterator first, Iterator last, int k) {
  size_t size = std::distance(first, last);
  std::vector<std::remove_reference_t<decltype(*first)>> buffer(size);
  std::vector<std::remove_reference_t<decltype(*first)>> array(size);

  size_t index = 0;
  for (Iterator p = first; p != last; ++p, ++index) {
    array[index] = *p;
  }

  return InternalDerandQuickSelect(array.begin(), array.end(), k,
                                   buffer.begin());
}

int GetNumber(int previous, int preprevious, int mod) {
  long long up = previous * 123 + preprevious * 45;
  return up % mod;
}

void FillSequence(std::vector<int>& seq) {
  int mod = 1e7 + 4321;
  for (size_t i = 2; i < seq.size(); ++i) {
    seq[i] = GetNumber(seq[i - 1], seq[i - 2], mod);
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(0);
  std::cout.precision(20);
  std::srand((unsigned int)time(NULL));

  size_t n = 0, k = 0;
  std::cin >> n >> k;

  std::vector<int> seq(n);
  std::cin >> seq[0] >> seq[1];
  FillSequence(seq);

  std::cout << DerandQuickSelect(seq.begin(), seq.end(), k);

  return 0;
}
