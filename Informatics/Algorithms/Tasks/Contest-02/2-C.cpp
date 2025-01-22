/*
C (3 балла, с ревью). Уменьшение ключа

Ограничение времени	1.5 секунд
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Реализуйте двоичную кучу.
Обработайте запросы следующих видов: insert x — вставить целое число x в кучу;
getMin — вывести значение минимального элемента в куче (гарантируется, что к
этому моменту куча не пуста); extractMin — удалить минимальный элемент из кучи,
выводить его не нужно (гарантируется, что к этому моменту куча не пуста);
decreaseKey i Δ — уменьшить число, вставленное на i-м запросе, на целое число Δ
≥ 0 (гарантируется, что i-й запрос был осуществлён ранее, являлся запросом
добавления, а добавленное на этом шаге число всё ещё лежит в куче). Обратите
внимание, число i равно номеру запроса среди всех запросов, а не только среди
запросов добавления! Можете считать, что в любой момент времени все числа,
хранящиеся в куче, попарно различны, а их количество не превышает 10^5.

Формат ввода
В первой строке содержится число q (1 ≤ q ≤ 10^6), означающее число запросов.
В следующих q строках содержатся запросы в описанном выше формате. Добавляемые
числа x и поправки Δ лежат в промежутке [−10^9, 10^9], а Δ ≥ 0.

Формат вывода
На каждый запрос вида getMin выведите ответ в отдельной строке.
*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <algorithm>
#include <iostream>
#include <vector>

namespace NT {
template <typename Type>
class Heap {
 public:
  Heap() : array_(1, 0), ptr_(1, 0), num_(1, 0){};
  ~Heap() {}

  void Insert(Type x);
  void ExtractMin();
  Type GetMin();
  void DecreaseKey(size_t key, Type val);

 private:
  std::vector<Type> array_;
  std::vector<size_t> ptr_;
  std::vector<size_t> num_;

  void Swap(size_t u, size_t v);
  void SiftUp(size_t v);
  void SiftDown(size_t v);
};

template <typename Type>
void Heap<Type>::Insert(Type x) {
  array_.push_back(x);

  ptr_.push_back(array_.size() - 1);
  num_.push_back(ptr_.size() - 1);
  SiftUp(array_.size() - 1);
}

template <typename Type>
void Heap<Type>::ExtractMin() {
  Swap(1, array_.size() - 1);

  array_.pop_back();
  num_.pop_back();
  SiftDown(1);
}

template <typename Type>
Type Heap<Type>::GetMin() {
  return array_[1];
}

template <typename Type>
void Heap<Type>::DecreaseKey(size_t key, Type val) {
  array_[ptr_[key]] -= val;
  SiftUp(ptr_[key]);
}

template <typename Type>
void Heap<Type>::Swap(size_t u, size_t v) {
  std::swap(array_[u], array_[v]);
  std::swap(ptr_[num_[u]], ptr_[num_[v]]);
  std::swap(num_[u], num_[v]);
}

template <typename Type>
void Heap<Type>::SiftUp(size_t v) {
  size_t v_new = v / 2;

  while (v != 1 && array_[v] < array_[v_new]) {
    Swap(v, v_new);
    v = v_new;
    v_new = v / 2;
  }
}

template <typename Type>
void Heap<Type>::SiftDown(size_t v) {
  size_t v_new = 2 * v;

  while (v_new < array_.size()) {
    size_t u = v_new +
               (v_new + 1 < array_.size() && array_[v_new] > array_[v_new + 1]);

    if (array_[v] <= array_[u]) {
      break;
    }

    Swap(u, v);
    v = u;
    v_new = 2 * v;
  }
}
}  // namespace NT

void HeapQuery(size_t n) {
  NT::Heap<long long> heap;
  std::vector<size_t> pos_adds(n);

  size_t num_adds = 0;
  for (size_t i = 0; i < n; ++i) {
    std::string s;
    std::cin >> s;

    if (s == "insert") {
      long long new_num = 0;
      std::cin >> new_num;

      heap.Insert(new_num);
      pos_adds[i] = ++num_adds;
    } else if (s == "getMin") {
      std::cout << heap.GetMin() << '\n';
    } else if (s == "extractMin") {
      heap.ExtractMin();
    } else if (s == "decreaseKey") {
      size_t key = 0;
      long long val = 0;
      std::cin >> key >> val;

      heap.DecreaseKey(pos_adds[key - 1], val);
    }
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(0), std::cout.tie(0);
  std::cout.precision(20);

  size_t n = 0;
  std::cin >> n;

  HeapQuery(n);

  return 0;
}
