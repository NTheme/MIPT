/*
G (2 балла, с ревью). Long long LSD

Ограничение времени	2 секунды
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Дан массив неотрицательных целых 64-битных чисел. Количество чисел не больше
10^6. Отсортировать массив методом поразрядной сортировки LSD по байтам.

Формат ввода
В первой строке вводится количество чисел в массиве N. Далее идут на N строках N
чисел.

Формат вывода
Выведите этот массив, отсортированный в порядке неубывания.
*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <iostream>
#include <vector>

const size_t kBlockSize = 255;

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

template <typename Type>
Type GetByte(Type val, size_t shift) {
  return val >> (shift * sizeof(val)) & kBlockSize;
}

template <typename Iterator>
void CountByte(Iterator first, Iterator last, size_t byte, Iterator byte_count,
               Iterator buf_array) {
  size_t size = std::distance(first, last);

  for (size_t i = 0; i < size; i++) {
    byte_count[GetByte(first[i], byte)]++;
  }
  for (size_t i = 1; i < 256; i++) {
    byte_count[i] += byte_count[i - 1];
  }
  for (int i = size - 1; i >= 0; --i) {
    buf_array[--byte_count[GetByte(first[i], byte)]] = first[i];
  }

  std::move(buf_array, buf_array + size, first);
}

template <typename Iterator>
void LSDSort(Iterator first, Iterator last) {
  size_t size = std::distance(first, last);

  size_t index = 0;
  std::vector<typename std::remove_reference<decltype(*first)>::type> array(
      size);
  for (Iterator p = first; p != last; ++p, ++index) {
    array[index] = *p;
  }

  std::vector<typename std::remove_reference<decltype(*first)>::type>
      byte_count(256);
  std::vector<typename std::remove_reference<decltype(*first)>::type> buf_array(
      size);

  size_t size_el =
      sizeof(typename std::remove_reference<decltype(*first)>::type);
  for (size_t byte = 0; byte < size_el; ++byte) {
    byte_count.assign(256, 0);
    buf_array.assign(size_el, 0);
    CountByte(array.begin(), array.end(), byte, byte_count.begin(),
              buf_array.begin());
  }

  index = 0;
  for (Iterator p = first; p != last; ++p, ++index) {
    *p = array[index];
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(0), std::cout.tie(0);
  std::cout.precision(20);
  std::srand((unsigned int)time(NULL));

  size_t n = 0;
  std::cin >> n;

  std::vector<long long> arr(n);
  std::cin >> arr;

  LSDSort(arr.begin(), arr.end());
  std::cout << arr;

  return 0;
}
