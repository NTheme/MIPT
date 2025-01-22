/*

C (1 балл, с ревью). А лучше там, где нас нет

Ограничение времени	1 секунда
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Он побывал во многих местах. В одном невиданный рассвет, в другом — море и
рубиновый закат. И он оценил каждое из мест, в a_i очков. Вам нужно уметь
отвечать всего лишь на два запроса: set(i, x) — присвоить i-му месту рейтинг x,
get(i, x) — найти ближайшую справа (в массиве рейтинга мест) локацию от i-й
(включая ее саму) с числом очков не меньшим, чем x. Надо вывести ее индекс.

Формат ввода
На первой строке число мест, где был он, n (1 ≤ n ≤ 2 * 10^5) и количество
запросов m (1 ≤ m ≤ 2 * 10^5). На второй строке n целых чисел — массив с очками
локаций a, они нумеруются в 1-индексации. Следующие m строк содержат запросы.
Запрос типа set: «0 i x».
Запрос типа get: «1 i x».
Все числа в вводе не превосходят двухсот тысяч.

Формат вывода
На каждой запрос типа «get» на отдельной строке выведите нужный индекс. Если
такого k не существует, выведите −1.

*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <algorithm>
#include <iostream>
#include <vector>

template <typename TypeLeft, typename TypeRight>
std::istream& operator>>(std::istream& inp,
                         std::pair<TypeLeft, TypeRight>& pair) {
  inp >> pair.first >> pair.second;
  return inp;
}
template <typename TypeLeft, typename TypeRight>
std::ostream& operator<<(std::ostream& out,
                         const std::pair<TypeLeft, TypeRight>& pair) {
  out << pair.first << ' ' << pair.second;
  return out;
}

template <typename Type>
std::istream& operator>>(std::istream& inp, std::vector<Type>& array) {
  for (auto& p : array) {
    inp >> p;
  }
  return inp;
}
template <typename Type>
std::ostream& operator<<(std::ostream& out, const std::vector<Type>& array) {
  for (const auto& p : array) {
    out << p << ' ';
  }
  return out;
}

struct Segment {
  size_t left;
  size_t right;

  Segment();
  Segment(size_t left_val, size_t right_val);
  ~Segment();

  Segment& operator=(const Segment& other);
  bool operator==(const Segment& other) const;

  size_t Middle() const;
  size_t Length() const;

  Segment LeftHalf() const;
  Segment RightHalf() const;
};

Segment::Segment() : left(0), right(0) {}

Segment::Segment(size_t left_val, size_t right_val)
    : left(left_val), right(right_val) {}

Segment::~Segment() {}

Segment& Segment::operator=(const Segment& other) {
  left = other.left;
  right = other.right;
  return *this;
}

bool Segment::operator==(const Segment& other) const {
  return left == other.left && right == other.right;
}

size_t Segment::Middle() const { return (left + right) / 2; }

size_t Segment::Length() const { return (right >= left) ? right - left : 0; }

Segment Segment::LeftHalf() const { return Segment(left, Middle()); }

Segment Segment::RightHalf() const { return Segment(Middle(), right); }

template <typename Type>
class SegmentTree {
 public:
  SegmentTree();
  SegmentTree(size_t size);
  SegmentTree(const std::vector<Type>& array);
  ~SegmentTree();

  void Build(const std::vector<Type>& array);
  Type Get(size_t left, Type request) const;
  void Update(size_t index, Type new_value);

  size_t Size() const;

 private:
  size_t size_;
  std::vector<Type> tree_;

  Type Merge(Type left, Type right) const;

  void BuildInternal(const std::vector<Type>& array, size_t vertex,
                     Segment tree_seg);
  void UpdateInternal(size_t vertex, Segment tree_seg, size_t index,
                      Type new_value);

  Type GetIndex(size_t vertex, Segment tree_seg, Type request) const;
  Type GetLeft(size_t vertex, Segment tree_seg, Segment query_seg,
               Type request) const;
  Type GetRight(size_t vertex, Segment tree_seg, Segment query_seg,
                Type request) const;
  Type GetInternal(size_t vertex, Segment tree_seg, Segment query_seg,
                   Type request) const;
};

template <typename Type>
SegmentTree<Type>::SegmentTree() : size_(0) {}

template <typename Type>
SegmentTree<Type>::SegmentTree(size_t size) : size_(size), tree_(4 * size_) {}

template <typename Type>
SegmentTree<Type>::SegmentTree(const std::vector<Type>& array) {
  Build(array);
}

template <typename Type>
SegmentTree<Type>::~SegmentTree() {}

template <typename Type>
void SegmentTree<Type>::Build(const std::vector<Type>& array) {
  size_ = array.size();
  tree_.assign(4 * size_, 0);
  BuildInternal(array, 1, Segment(0, size_));
}

template <typename Type>
Type SegmentTree<Type>::Get(size_t left, Type request) const {
  return GetInternal(1, Segment(0, size_), Segment(left, size_), request);
}

template <typename Type>
void SegmentTree<Type>::Update(size_t index, Type new_value) {
  UpdateInternal(1, Segment(0, size_), index, new_value);
}

template <typename Type>
size_t SegmentTree<Type>::Size() const {
  return size_;
}

template <typename Type>
Type SegmentTree<Type>::Merge(Type left, Type right) const {
  return std::max(left, right);
}

template <typename Type>
void SegmentTree<Type>::BuildInternal(const std::vector<Type>& array,
                                      size_t vertex, Segment tree_seg) {
  if (tree_seg.Length() == 1) {
    tree_[vertex] = array[tree_seg.left];
    return;
  }

  BuildInternal(array, 2 * vertex, tree_seg.LeftHalf());
  BuildInternal(array, 2 * vertex + 1, tree_seg.RightHalf());
  tree_[vertex] = Merge(tree_[2 * vertex], tree_[2 * vertex + 1]);
}

template <typename Type>
void SegmentTree<Type>::UpdateInternal(size_t vertex, Segment tree_seg,
                                       size_t index, Type new_value) {
  if (tree_seg.Length() == 1) {
    tree_[vertex] = new_value;
    return;
  }

  if (index < tree_seg.Middle()) {
    UpdateInternal(2 * vertex, tree_seg.LeftHalf(), index, new_value);
  } else {
    UpdateInternal(2 * vertex + 1, tree_seg.RightHalf(), index, new_value);
  }
  tree_[vertex] = Merge(tree_[2 * vertex], tree_[2 * vertex + 1]);
}

template <typename Type>
Type SegmentTree<Type>::GetIndex(size_t vertex, Segment tree_seg,
                                 Type request) const {
  if (tree_[vertex] >= request) {
    while (tree_seg.Length() > 1) {
      vertex *= 2;
      if (tree_[vertex] >= request) {
        tree_seg = tree_seg.LeftHalf();
      } else {
        tree_seg = tree_seg.RightHalf();
        ++vertex;
      }
    }

    return tree_seg.right;
  }
  return -1;
}

template <typename Type>
Type SegmentTree<Type>::GetLeft(size_t vertex, Segment tree_seg,
                                Segment query_seg, Type request) const {
  Segment query_new(query_seg.left,
                    std::min(query_seg.right, tree_seg.Middle()));
  return GetInternal(2 * vertex, tree_seg.LeftHalf(), query_new, request);
}

template <typename Type>
Type SegmentTree<Type>::GetRight(size_t vertex, Segment tree_seg,
                                 Segment query_seg, Type request) const {
  Segment query_new(std::max(query_seg.left, tree_seg.Middle()),
                    query_seg.right);
  return GetInternal(2 * vertex + 1, tree_seg.RightHalf(), query_new, request);
}

template <typename Type>
Type SegmentTree<Type>::GetInternal(size_t vertex, Segment tree_seg,
                                    Segment query_seg, Type request) const {
  if (query_seg.Length() == 0) {
    return -1;
  }
  if (tree_seg == query_seg) {
    return GetIndex(vertex, tree_seg, request);
  }

  Type left_value = GetLeft(vertex, tree_seg, query_seg, request);
  if (left_value != -1) {
    return left_value;
  }

  return GetRight(vertex, tree_seg, query_seg, request);
}

template <typename Type>
std::istream& operator>>(std::istream& inp, SegmentTree<Type>& tree) {
  std::vector<Type> array(tree.Size());
  inp >> array;
  tree.Build(array);
  return inp;
}

void ProcessQuery(SegmentTree<int>& tree, size_t type, size_t index,
                  size_t value) {
  if (type == 0) {
    tree.Update(index - 1, value);
  } else if (type == 1) {
    std::cout << tree.Get(index - 1, value) << '\n';
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t n = 0, m = 0;
  std::cin >> n >> m;

  SegmentTree<int> tree(n);
  std::cin >> tree;

  while (m-- > 0) {
    size_t type = 2, index = 0, value = 0;
    std::cin >> type >> index >> value;
    ProcessQuery(tree, type, index, value);
  }

  return 0;
}
