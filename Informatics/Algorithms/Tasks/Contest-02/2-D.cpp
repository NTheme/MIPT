/*
D (2 балла). Первые k элементов длинной последовательности

Ограничение времени	1 секунда
Ограничение памяти	16Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Дана длинная последовательность целых чисел длины n. Требуется вывести в
отсортированном виде её наименьшие k элементов. Чтобы не пришлось считывать
большую последовательность, её элементы задаются формулой. А именно, во входных
данных содержатся числа a_0, x, y. Тогда a_i = (x * a_(i − 1) + y) (mod 2^30).
Искомая последовательность — a_1, a_2, ..., a_n. Обратите внимание на
ограничение по памяти.

Формат ввода
В первой строке записаны n и k (1 ≤ n ≤ 10^7, 1 ≤ k ≤ 1000). В следующей строке
через пробел заданы значения a_0, x, y (0 ≤ a_0, x, y < 2^30).

Формат вывода
Выведите k наименьших элементов последовательности в отсортированном виде.
*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

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
    out << p << ' ';
  }
  return out;
}

namespace NT {
template <typename Type>
struct Heap {
 public:
  Heap() : array_(1, 0), size_(0){};

  ~Heap() {}

  void SiftUp(size_t v) {
    while (v != 1 && array_[v] > array_[v / 2]) {
      Swap(array_.begin() + v, array_.begin() + v / 2);
      v /= 2;
    }
  }

  void SiftDown(size_t v) {
    while (2 * v <= size_) {
      size_t u = 2 * v;
      if (2 * v + 1 <= size_ && array_[2 * v] < array_[2 * v + 1]) {
        u = 2 * v + 1;
      }
      if (array_[v] >= array_[u]) {
        break;
      }
      Swap(array_.begin() + u, array_.begin() + v);
      v = u;
    }
  }

  void Insert(int x) {
    array_.push_back(x);
    ++size_;

    SiftUp(size_);
  }

  void ExtractMin() {
    Swap(array_.begin() + 1, array_.begin() + size_--);
    array_.pop_back();

    SiftDown(1);
  }

  int GetMin() { return array_[1]; }

  size_t Size() { return size_; }

 private:
  std::deque<int> array_;
  size_t size_;

  template <typename TypeIterator>
  static void Swap(TypeIterator a, TypeIterator b) {
    Type tmp = *a;
    *a = *b;
    *b = tmp;
  }
};
}  // namespace NT

long long GetNumber(long long x, long long y, long long mod, long long prev) {
  return (x * prev + y) % mod;
}

NT::Heap<int> GetMinK(size_t n, size_t k, long long x, long long y,
                      long long last) {
  long long mod = 1 << 30;

  NT::Heap<int> heap;

  for (size_t i = 0; i < n; ++i) {
    last = GetNumber(x, y, mod, last);

    if (heap.Size() < k) {
      heap.Insert(last);
    } else if (heap.GetMin() > last) {
      heap.ExtractMin();
      heap.Insert(last);
    }
  }

  return heap;
}

std::deque<int> ConvertToAnswer(NT::Heap<int>& heap) {
  size_t k = heap.Size();

  std::deque<int> kelem;
  for (size_t i = 0; i < k; ++i) {
    kelem.push_front(heap.GetMin());
    heap.ExtractMin();
  }

  return kelem;
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(0), std::cout.tie(0);
  std::cout.precision(20);
  std::srand((unsigned int)time(NULL));

  size_t n = 0, k = 0;
  std::cin >> n >> k;

  long long last = 0, x = 0, y = 0;
  std::cin >> last >> x >> y;

  NT::Heap<int> heap = GetMinK(n, k, x, y, last);
  std::deque<int> kelem = ConvertToAnswer(heap);

  std::cout << kelem;

  return 0;
}
