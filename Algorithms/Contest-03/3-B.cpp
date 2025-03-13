/*
B (2 балла). Эпизод 2. Скрытые дизлайки.

Ограничение времени	2 секунды
Ограничение памяти	64Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

В связи с тем, что на ютубе скрыли число дизлайков, Мирону необходимо понимать,
насколько его творчество заходит аудитории. Данные о числе реакций обновляются
не в режиме онлайн, а каждый час. При этом неизвестно, что из них лайки, а что
из них дизлайки. То есть задан массив a_1, …, a_n — число реакций, пришедшее в
каждый час. Исполнитель уверен, что лайки и дизлайки чередуются, поэтому он
считает поддержку за промежуток времени с l-го по r-й час по формуле (a_l − a_(l
+ 1) + a_(l + 2) − … ± a_r). Более того, он хочет смотреть, как меняется его
восприятие, если в какой-то момент пришло бы другое число реакций.

Формат ввода
В первой строке входного файла содержится натуральное число n (1 ≤ n ≤ 10^5) —
длина массива с почасовыми реакциями. Во второй строке записаны начальные
значения элементов (неотрицательные целые числа, не превосходящие 10^4).
В третьей строке находится натуральное число m (1 ≤ m ≤ 10^5) — количество
запросов исполнителя. В последующих m строках записаны операции: операция
изменения значения задается тремя числами 0 i j (1 ≤ i ≤ n, 1 ≤ j ≤ 10^4).
Формально, a_i = j. Операция запроса поддержки задается тремя числами 1 l r (1 ≤
l ≤ r ≤ n).

Формат вывода
Для каждой операции второго типа выведите на отдельной строке соответствующую
знакочередующуюся сумму.
*/

#include <algorithm>
#include <iostream>
#include <vector>

template <typename T1, typename T2>
std::istream& operator>>(std::istream& inp, std::pair<T1, T2>& pair) {
  inp >> pair.first >> pair.second;
  return inp;
}
template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& out, const std::pair<T1, T2>& pair) {
  out << pair.first << ' ' << pair.second;
  return out;
}

template <typename T>
std::istream& operator>>(std::istream& inp, std::vector<T>& arr) {
  for (auto& p : arr) {
    inp >> p;
  }
  return inp;
}
template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& arr) {
  for (const auto& p : arr) {
    out << p << ' ';
  }
  return out;
}

struct Segment {
  size_t left;
  size_t right;

  Segment(size_t left_val, size_t right_val);

  Segment& operator=(const Segment& other);
  bool operator==(const Segment& other) const;

  size_t Middle() const;
};

Segment::Segment(size_t left_val, size_t right_val)
    : left(left_val), right(right_val) {}

Segment& Segment::operator=(const Segment& other) {
  left = other.left;
  right = other.right;
  return *this;
}

bool Segment::operator==(const Segment& other) const {
  return left == other.left && right == other.right;
}

size_t Segment::Middle() const { return (left + right) / 2; }

template <typename Type>
struct Vertex {
  Type value;
  size_t num_elem;

  Vertex(Type value_val, size_t num_elem_val)
      : value(value_val), num_elem(num_elem_val) {}
  Vertex(const Vertex& left, const Vertex& right)
      : value(left.value + right.value * ((left.num_elem % 2 == 0) ? 1 : -1)),
        num_elem(left.num_elem + right.num_elem) {}
};

template <typename Type>
class SegTree {
 public:
  SegTree();
  SegTree(std::vector<Type>& array);
  ~SegTree() {}

  void Build(std::vector<Type>& array);
  Type Get(Segment query_seg);
  void Update(size_t elem, Type val);

  size_t Size() const;

 private:
  size_t size_;
  std::vector<Vertex<Type>> tree_;

  const size_t kCoeffTree = 4;

  void BuildInternal(std::vector<Type>& array, size_t vertex, Segment tree_seg);
  Vertex<Type> GetInternal(size_t vertex, Segment tree_seg, Segment query_seg);
  void UpdateInternal(size_t vertex, Segment tree_seg, size_t elem, Type val);
};

template <typename Type>
SegTree<Type>::SegTree() : size_(0) {}

template <typename Type>
SegTree<Type>::SegTree(std::vector<Type>& array) {
  Build(array);
}

template <typename Type>
void SegTree<Type>::Build(std::vector<Type>& array) {
  size_ = array.size();
  tree_.assign(kCoeffTree * size_, Vertex(0, 0));
  BuildInternal(array, 1, Segment(0, size_));
}

template <typename Type>
Type SegTree<Type>::Get(Segment query_seg) {
  return GetInternal(1, Segment(0, size_), query_seg).value;
}

template <typename Type>
void SegTree<Type>::Update(size_t elem, Type val) {
  UpdateInternal(1, Segment(0, size_), elem, val);
}

template <typename Type>
size_t SegTree<Type>::Size() const {
  return size_;
}

template <typename Type>
void SegTree<Type>::BuildInternal(std::vector<Type>& array, size_t vertex,
                                  Segment tree_seg) {
  if (tree_seg.left + 1 == tree_seg.right) {
    tree_[vertex] = Vertex<Type>(array[tree_seg.left], 1);
    return;
  }
  BuildInternal(array, 2 * vertex, Segment(tree_seg.left, tree_seg.Middle()));
  BuildInternal(array, 2 * vertex + 1,
                Segment(tree_seg.Middle(), tree_seg.right));
  tree_[vertex] = Vertex<Type>(tree_[2 * vertex], tree_[2 * vertex + 1]);
}

template <typename Type>
Vertex<Type> SegTree<Type>::GetInternal(size_t vertex, Segment tree_seg,
                                        Segment query_seg) {
  if (query_seg.left >= query_seg.right) {
    return Vertex<Type>(0, 0);
  }
  if (tree_seg == query_seg) {
    return tree_[vertex];
  }

  return Vertex<Type>(
      GetInternal(2 * vertex, Segment(tree_seg.left, tree_seg.Middle()),
                  Segment(query_seg.left,
                          std::min(query_seg.right, tree_seg.Middle()))),
      GetInternal(2 * vertex + 1, Segment(tree_seg.Middle(), tree_seg.right),
                  Segment(std::max(query_seg.left, tree_seg.Middle()),
                          query_seg.right)));
}

template <typename Type>
void SegTree<Type>::UpdateInternal(size_t vertex, Segment tree_seg, size_t elem,
                                   Type val) {
  if (tree_seg.left + 1 == tree_seg.right) {
    tree_[vertex].value = val;
    return;
  }
  if (elem < tree_seg.Middle()) {
    UpdateInternal(2 * vertex, Segment(tree_seg.left, tree_seg.Middle()), elem,
                   val);
  } else {
    UpdateInternal(2 * vertex + 1, Segment(tree_seg.Middle(), tree_seg.right),
                   elem, val);
  }
  tree_[vertex] = Vertex(tree_[2 * vertex], tree_[2 * vertex + 1]);
}

void ProcessQueries(std::vector<int>& start, size_t q) {
  SegTree<int> segtree(start);

  while (q-- > 0) {
    size_t type = 2;
    std::cin >> type;

    if (type == 0) {
      size_t i, j;
      std::cin >> i >> j;
      segtree.Update(i - 1, j);
    } else {
      size_t l, r;
      std::cin >> l >> r;
      std::cout << segtree.Get(Segment(l - 1, r)) << '\n';
    }
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(6);

  size_t n = 0, q = 0;
  std::cin >> n;

  std::vector<int> start(n);
  std::cin >> start >> q;

  ProcessQueries(start, q);

  return 0;
}
