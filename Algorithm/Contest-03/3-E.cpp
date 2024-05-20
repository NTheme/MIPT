/*
E (3 балла). Очередная странная структура

Ограничение времени	1 секунда
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Нужно отвечать на запросы вида
+ x — добавить в мультимножество число x.
? x — посчитать сумму чисел не больших x.

Формат ввода
В первой строке содержится число запросов 1 ≤ q ≤ 10^5. Далее каждая строка
содержит один запрос. Все числа x целые от 0 до 10^9 − 1.

Формат вывода
Ответы на все запросы вида ? x.
*/

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
  Type Get(size_t right) const;
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
  Type GetInternal(size_t vertex, Segment tree_seg, Segment query_seg) const;
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
Type SegmentTree<Type>::Merge(Type left, Type right) const {
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
std::istream& operator>>(std::istream& inp, SegmentTree<Type>& tree) {
  std::vector<Type> array(tree.Size());
  inp >> array;
  tree.Build(array);
  return inp;
}

void MakeArray(const std::vector<std::pair<char, size_t>>& queries,
               std::vector<size_t>& sorted, std::vector<size_t>& indexes) {
  std::vector<std::pair<size_t, size_t>> count_indexes;
  for (const auto& query : queries) {
    if (query.first == '+') {
      count_indexes.push_back({query.second, count_indexes.size()});
    }
  }
  std::sort(count_indexes.begin(), count_indexes.end());

  for (const auto& elem : count_indexes) {
    sorted.push_back(elem.first);
  }

  for (size_t index = 0; index < count_indexes.size(); ++index) {
    count_indexes[index] = {count_indexes[index].second, index};
  }
  std::sort(count_indexes.begin(), count_indexes.end());

  for (const auto& elem : count_indexes) {
    indexes.push_back(elem.second);
  }
}

void ProcessQueries(std::vector<std::pair<char, size_t>>& queries) {
  std::vector<size_t> sorted, indexes;
  MakeArray(queries, sorted, indexes);

  SegmentTree<size_t> tree(std::vector<size_t>(indexes.size(), 0));

  size_t cur_add = 0;
  for (const auto& query : queries) {
    if (query.first == '+') {
      tree.Update(indexes[cur_add++], query.second);
    } else if (query.first == '?') {
      size_t pp = std::upper_bound(sorted.begin(), sorted.end(), query.second) -
                  sorted.begin();
      std::cout << tree.Get(pp) << '\n';
    }
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t q = 0;
  std::cin >> q;

  std::vector<std::pair<char, size_t>> queries(q);
  std::cin >> queries;

  ProcessQueries(queries);

  return 0;
}
