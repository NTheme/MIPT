/*
E (4 балла, с ревью). Минимакс

Ограничение времени	2 секунды
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Реализуйте структуру данных, способную выполнять операции ниже. Напишите
программу, реализовав все указанные здесь методы. Возможные команды для
программы:

insert n — добавить в структуру число n (1 ≤ n ≤ 10^9) (значение n задается
после команды). Программа должна вывести ok. extract_min — удалить из структуры
минимальный элемент. Программа должна вывести его значение.
get_min — программа должна вывести значение минимального элемента, не удаляя его
из структуры. extract_max — удалить из структуры максимальный элемент. Программа
должна вывести его значение. get_max — программа должна вывести значение
миаксимального элемента, не удаляя его из структуры. size — программа должна
вывести количество элементов в структуре. clear — Программа должна очистить
структуру и вывести ok. Перед исполнением операций extract_min, extract_max,
get_min и get_max программа должна проверять, содержится ли в структуре хотя бы
один элемент.

Если во входных данных встречается операция extract_min, extract_max, get_min
или get_max, и при этом в структуре нет ни одного элемента, то программа должна
вместо числового значения вывести строку error.

Формат ввода
В первой строке входных данных записано единственное число M (1 ≤ M ≤ 2 ⋅ 105) —
количество команд. В следующих М строках дано по одной команде из тех, что идут
выше.

Формат вывода
Для каждой команды выведите одну строчку — результат ее выполнения.
*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <iostream>
#include <vector>

template <typename Type>
std::istream& operator>>(std::istream& inp, std::vector<Type>& arr) {
  for (auto& p : arr) {
    inp >> p;
  }
  return inp;
}
template <typename Type>
std::ostream& operator<<(std::ostream& out, std::vector<Type> arr) {
  for (auto p : arr) {
    out << p << ' ';
  }
  return out;
}

namespace NT {
template <typename Type>
class Heap {
 public:
  typedef enum { HeapMax, HeapMin } HeapType;

  Heap(HeapType type)
      : array_(1, 0), ptr_(1, 0), num_(1, 0), heap_type_(type){};
  ~Heap() {}

  void Insert(Type x);
  size_t ExtractExtr();
  Type EraseElem(size_t query);
  Type GetExtr();
  size_t Size();

 private:
  std::vector<Type> array_;
  std::vector<size_t> ptr_;
  std::vector<size_t> num_;

  HeapType heap_type_;

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
size_t Heap<Type>::ExtractExtr() {
  if (array_.size() == 1) {
    return 0;
  }

  size_t query = num_[1];
  EraseElem(query);

  return query;
}

template <typename Type>
Type Heap<Type>::EraseElem(size_t query) {
  if (array_.size() == 1) {
    return 0;
  }

  size_t position = ptr_[query];
  Type val = array_[position];
  Swap(position, array_.size() - 1);

  array_.pop_back();
  num_.pop_back();
  SiftDown(position);
  SiftUp(position);

  return val;
}

template <typename Type>
Type Heap<Type>::GetExtr() {
  if (array_.size() == 1) {
    return 0;
  }
  return array_[1];
}

template <typename Type>
size_t Heap<Type>::Size() {
  return array_.size() - 1;
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

  while (v != 1 && 0 < (array_[v_new] - array_[v]) * (heap_type_ ? 1 : -1)) {
    Swap(v, v_new);

    v = v_new;
    v_new = v / 2;
  }
}

template <typename Type>
void Heap<Type>::SiftDown(size_t v) {
  size_t v_new = 2 * v;

  while (v_new < array_.size()) {
    size_t u = v_new;
    u += (v_new + 1 < array_.size() &&
          0 > (array_[v_new + 1] - array_[v_new]) * (heap_type_ ? 1 : -1));

    if (0 <= (array_[u] - array_[v]) * (heap_type_ ? 1 : -1)) {
      break;
    }

    Swap(u, v);
    v = u;
    v_new = 2 * v;
  }
}

template <typename Type>
class MinMaxHeap {
 public:
  MinMaxHeap() : min_(Heap<Type>::HeapMin), max_(Heap<Type>::HeapMax) {}
  ~MinMaxHeap() {}

  void Insert(Type x);
  Type GetMin();
  Type GetMax();
  Type ExtractMin();
  Type ExtractMax();
  size_t Size();
  void Clear();

 private:
  Heap<Type> min_;
  Heap<Type> max_;
};

template <typename Type>
void MinMaxHeap<Type>::Insert(Type x) {
  min_.Insert(x);
  max_.Insert(x);
}

template <typename Type>
Type MinMaxHeap<Type>::GetMin() {
  return min_.GetExtr();
}

template <typename Type>
Type MinMaxHeap<Type>::GetMax() {
  return max_.GetExtr();
}

template <typename Type>
Type MinMaxHeap<Type>::ExtractMin() {
  return max_.EraseElem(min_.ExtractExtr());
}

template <typename Type>
Type MinMaxHeap<Type>::ExtractMax() {
  return min_.EraseElem(max_.ExtractExtr());
}

template <typename Type>
size_t MinMaxHeap<Type>::Size() {
  return min_.Size();
}

template <typename Type>
void MinMaxHeap<Type>::Clear() {
  while (min_.Size() > 0) {
    ExtractMax();
  }
}
}  // namespace NT

void PrintValue(int val) {
  if (val < 1) {
    std::cout << "error\n";
  } else {
    std::cout << val << '\n';
  }
}

void TwoHeapQuery(NT::MinMaxHeap<int>& two_heap) {
  std::string s;
  std::cin >> s;

  if (s == "insert") {
    int n = 0;
    std::cin >> n;
    two_heap.Insert(n);
    std::cout << "ok\n";
  } else if (s == "extract_min") {
    PrintValue(two_heap.ExtractMin());
  } else if (s == "get_min") {
    PrintValue(two_heap.GetMin());
  } else if (s == "extract_max") {
    PrintValue(two_heap.ExtractMax());
  } else if (s == "get_max") {
    PrintValue(two_heap.GetMax());
  } else if (s == "size") {
    std::cout << two_heap.Size() << '\n';
  } else if (s == "clear") {
    two_heap.Clear();
    std::cout << "ok\n";
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(0);
  std::cout.precision(20);
  std::srand((unsigned int)time(NULL));

  size_t m = 0;
  std::cin >> m;

  NT::MinMaxHeap<int> two_heap;

  while (m-- > 0) {
    TwoHeapQuery(two_heap);
  }

  return 0;
}
