/*
F (4 балла). Возня на круге

Ограничение времени	1 секунда
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Представьте себе клетчатую окружность, состояющую из L клеток. Клетки нумеруются
целыми числами от 1 до L. Некоторые N клеток закрашены. Окружность можно
разрезать между любыми двумя клетками. Всего существует L различных разрезов.
Получившаяся полоска делится на K равных частей (L кратно K). Для каждого i
определим f_i как количество закрашенных клеток в i -й части. Вам нужно найти
такой разрез, что F = max (f_i) - min(f_j) минимально возможно. Кроме того, вам
нужно найти количество разрезов, на которых достигается минимум.

Формат ввода
На первой строке целые числа L, N, K, (K ≥ 2). Гарантируется, что L делится на
K. На второй строке N различных целых чисел от 1 до L — номера закрашенных
клеток.

Формат вывода
На первой строке выведите минимальное значение F и количество разрезов с таким
F. На второй строке выведите любое x от 1 до L, что, если разрезать окружность
между клетками x и x + 1, а после чего посчитать F, то получится минимальное
значение.

Ограничения: N, K ≤ 500000, L ≤ 10^18.
*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <algorithm>
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

namespace NT {
template <typename Type>
class Heap {
 public:
  Heap(bool type)
      : array_(1, 0),
        size_(0),
        added_(0),
        is_min_heap_(type ? 1 : -1),
        ptr_(1, 0),
        que_(1, 0){};
  ~Heap() {}

  void Insert(Type x);
  void ChangeKey(size_t queue, Type val);
  size_t ExtractExtr();
  Type EraseElem(size_t query);
  Type GetExtr();
  size_t Size();

 private:
  std::vector<Type> array_;
  size_t size_;
  size_t added_;
  int is_min_heap_;

  std::vector<size_t> ptr_;
  std::vector<size_t> que_;

  void Swap(size_t u, size_t v);
  void SiftUp(size_t v);
  void SiftDown(size_t v);
};

template <typename Type>
void Heap<Type>::Insert(Type x) {
  array_.push_back(x);
  ++size_, ++added_;

  ptr_.push_back(size_);
  que_.push_back(added_);

  SiftUp(size_);
}

template <typename Type>
void Heap<Type>::ChangeKey(size_t queue, Type val) {
  array_[ptr_[queue]] += val;
  SiftDown(ptr_[queue]);
  SiftUp(ptr_[queue]);
}

template <typename Type>
size_t Heap<Type>::ExtractExtr() {
  if (size_ == 0) {
    return 0;
  }

  size_t query = que_[1];
  EraseElem(query);

  return query;
}

template <typename Type>
Type Heap<Type>::EraseElem(size_t query) {
  if (size_ == 0) {
    return 0;
  }

  size_t position = ptr_[query];
  Type val = array_[position];
  Swap(position, size_--);

  array_.pop_back();
  que_.pop_back();

  SiftDown(position);
  SiftUp(position);

  return val;
}

template <typename Type>
Type Heap<Type>::GetExtr() {
  if (size_ == 0) {
    return 0;
  }
  return array_[1];
}

template <typename Type>
size_t Heap<Type>::Size() {
  return size_;
}

template <typename Type>
void Heap<Type>::Swap(size_t u, size_t v) {
  std::swap(array_[u], array_[v]);
  std::swap(ptr_[que_[u]], ptr_[que_[v]]);
  std::swap(que_[u], que_[v]);
}

template <typename Type>
void Heap<Type>::SiftUp(size_t v) {
  size_t v_new = v / 2;

  while (v != 1 && 0 < (array_[v_new] - array_[v]) * is_min_heap_) {
    Swap(v, v_new);

    v = v_new;
    v_new = v / 2;
  }
}

template <typename Type>
void Heap<Type>::SiftDown(size_t v) {
  size_t v_new = 2 * v;

  while (v_new <= size_) {
    size_t u = v_new;
    if (v_new + 1 <= size_ &&
        0 > (array_[v_new + 1] - array_[v_new]) * is_min_heap_) {
      u = v_new + 1;
    }
    if (0 <= (array_[u] - array_[v]) * is_min_heap_) {
      break;
    }

    Swap(u, v);
    v = u;
    v_new = 2 * v;
  }
}

template <typename Type>
class TwoHeap {
 public:
  TwoHeap() : min_(true), max_(false) {}
  TwoHeap(const std::vector<Type>& array) : min_(true), max_(false) {
    for (auto p : array) {
      Insert(p);
    }
  }
  ~TwoHeap() {}

  void Insert(Type x);
  void ChangeKey(size_t queue, Type val);
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
void TwoHeap<Type>::Insert(Type x) {
  min_.Insert(x);
  max_.Insert(x);
}

template <typename Type>
void TwoHeap<Type>::ChangeKey(size_t queue, Type val) {
  min_.ChangeKey(queue, val);
  max_.ChangeKey(queue, val);
}

template <typename Type>
Type TwoHeap<Type>::GetMin() {
  return min_.GetExtr();
}

template <typename Type>
Type TwoHeap<Type>::GetMax() {
  return max_.GetExtr();
}

template <typename Type>
Type TwoHeap<Type>::ExtractMin() {
  return max_.EraseElem(min_.ExtractExtr());
}

template <typename Type>
Type TwoHeap<Type>::ExtractMax() {
  return min_.EraseElem(max_.ExtractExtr());
}

template <typename Type>
size_t TwoHeap<Type>::Size() {
  return min_.Size();
}

template <typename Type>
void TwoHeap<Type>::Clear() {
  while (min_.Size() > 0) {
    ExtractMax();
  }
}
}  // namespace NT

long long CheckAllRemains(std::vector<std::pair<long long, long long>>& remains,
                          NT::TwoHeap<long long>& segments, long long k,
                          long long& num_f, long long& pos_f) {
  long long min_f = -1;

  for (size_t pt = 0; pt < remains.size() - 1;) {
    long long cur_rest = remains[pt].first;
    while (pt < remains.size() - 1 && remains[pt].first == cur_rest) {
      segments.ChangeKey(remains[pt].second + 1, -1);
      segments.ChangeKey(remains[pt].second + (remains[pt].second != 0 ? 0 : k),
                         1);
      ++pt;
    }

    long long cur_f = segments.GetMax() - segments.GetMin();
    if (cur_f < min_f || min_f == -1) {
      min_f = cur_f;
      num_f = remains[pt].first - remains[pt - 1].first;
      pos_f = cur_rest + 1;
    } else if (cur_f == min_f) {
      num_f += remains[pt].first - remains[pt - 1].first;
    }
  }

  num_f *= k;

  return min_f;
}

long long GetMinF(long long l, long long k, std::vector<long long>& colored,
                  long long& num_f, long long& pos_f) {
  long long len_segment = l / k;

  std::vector<std::pair<long long, long long>> remains;
  for (auto& p : colored) {
    remains.push_back({(p - 1) % len_segment, (p - 1) / len_segment});
  }
  DerandQuickSort(remains.begin(), remains.end());

  std::vector<long long> col_segment(k);
  for (auto& p : remains) {
    col_segment[p.second]++;
  }

  NT::TwoHeap<long long> segments(col_segment);
  remains.emplace_back(remains[0].first + len_segment, remains[0].second);

  return CheckAllRemains(remains, segments, k, num_f, pos_f);
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(0);
  std::cout.precision(20);

  long long l = 0, n = 0, k = 0;
  std::cin >> l >> n >> k;

  std::vector<long long> colored(n);
  std::cin >> colored;

  long long num_f = 0;
  long long pos_f = 0;
  long long min_f = GetMinF(l, k, colored, num_f, pos_f);

  std::cout << min_f << ' ' << num_f << '\n' << pos_f << '\n';

  return 0;
}
