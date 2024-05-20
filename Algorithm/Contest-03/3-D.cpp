/*
D (2 балла). Переплетенные отрезки

Ограничение времени	2 секунды
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

В матрице был единый для всех ее вариаций архитектор. И он решил, что все n
Матриц расположены на одной прямой. А как нам известно, все переплетено. Будем
считать пару Матриц переплетенной, если их временные границы не совпадают, и
одна содержит целиком вторую. Посчитайте количество пар переплетенных Матриц.

Формат ввода
Целоы число n (1 ≤ n ≤ 300000) и n пар целых чисел 0 ≤ l_i ≤ r_i ≤ 10^9 —
временные границы Матриц.

Формат вывода
Одно число — количество пар вложенных Матриц.
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

std::istream& operator>>(std::istream& inp, Segment& seg) {
  inp >> seg.left >> seg.right;
  return inp;
}

template <typename Type>
class SegmentTree {
 public:
  SegmentTree();
  SegmentTree(size_t size);
  SegmentTree(const std::vector<Type>& array);
  ~SegmentTree() {}

  void Build(const std::vector<Type>& array);
  Type Get(size_t right) const;
  void Update(size_t index, Type new_value);

  size_t Size() const;

 private:
  size_t size_;
  std::vector<Type> tree_;

  Type Merge(Type left, Type right);
  void BuildInternal(const std::vector<Type>& array, size_t vertex,
                     Segment tree_seg);
  Type GetInternal(size_t vertex, Segment tree_seg, Segment query_seg) const;
  void UpdateInternal(size_t vertex, Segment tree_seg, size_t index,
                      Type new_value);
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
void SegmentTree<Type>::Build(const std::vector<Type>& array) {
  size_ = array.size();
  tree_.assign(4 * size_, 0);
  BuildInternal(array, 1, Segment(0, size_));
}

template <typename Type>
Type SegmentTree<Type>::Get(size_t right) const {
  return GetInternal(1, Segment(0, size_), Segment(0, right));
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
std::istream& operator>>(std::istream& inp, SegmentTree<Type>& tree) {
  std::vector<Type> array(tree.Size());
  inp >> array;
  tree.Build(array);
  return inp;
}

template <typename Type>
Type SegmentTree<Type>::Merge(Type left, Type right) {
  return left + right;
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
Type SegmentTree<Type>::GetInternal(size_t vertex, Segment tree_seg,
                                    Segment query_seg) const {
  if (query_seg.Length() == 0) {
    return 0;
  }
  if (tree_seg == query_seg) {
    return tree_[vertex];
  }

  Segment query_left(query_seg.left,
                     std::min(query_seg.right, tree_seg.Middle()));
  Segment query_right(std::max(query_seg.left, tree_seg.Middle()),
                      query_seg.right);
  return GetInternal(2 * vertex, tree_seg.LeftHalf(), query_left) +
         GetInternal(2 * vertex + 1, tree_seg.RightHalf(), query_right);
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

bool ComparatorRight(const std::pair<Segment, size_t>& left,
                     const std::pair<Segment, size_t>& right) {
  if (left.first.right < right.first.right) {
    return true;
  }
  if (left.first.right > right.first.right) {
    return false;
  }
  return left.first.left > right.first.left;
}

bool ComparatorLeft(const std::pair<Segment, size_t>& left,
                    const std::pair<Segment, size_t>& right) {
  if (left.first.left < right.first.left) {
    return true;
  }
  if (left.first.left > right.first.left) {
    return false;
  }
  if (left.first.right > right.first.right) {
    return true;
  }
  if (left.first.right < right.first.right) {
    return false;
  }
  return left.second < right.second;
}

void CountIndexes(std::vector<Segment>& segments,
                  std::vector<size_t>& borders) {
  std::vector<std::pair<Segment, size_t>> count_borders(segments.size());
  for (size_t index = 0; index < segments.size(); ++index) {
    count_borders[index] = {segments[index], index};
  }

  std::sort(count_borders.begin(), count_borders.end(), ComparatorRight);
  for (size_t index = 0; index < segments.size(); ++index) {
    count_borders[index].second = index;
  }
  std::sort(count_borders.begin(), count_borders.end(), ComparatorLeft);

  for (size_t index = 0; index < segments.size(); ++index) {
    borders[index] = count_borders[index].second;
  }
}

long long CountPairs(std::vector<Segment>& segments) {
  std::vector<size_t> borders(segments.size());
  CountIndexes(segments, borders);

  SegmentTree<size_t> right_borders(std::vector<size_t>(segments.size(), 1));

  long long num_pairs = 0;
  for (const auto& border : borders) {
    right_borders.Update(border, 0);
    num_pairs += right_borders.Get(border);
  }

  return num_pairs;
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t n = 0;
  std::cin >> n;

  std::vector<Segment> segments(n);
  std::cin >> segments;

  std::cout << CountPairs(segments) << '\n';

  return 0;
}
