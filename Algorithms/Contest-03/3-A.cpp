/*

A (2 балла). Марафон до нового альбома

Ограничение времени	1 секунда
Ограничение памяти	32Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

В этой задаче Марк бежит марафон до нового альбома. Вам предстоит написать
систему стимулирования его, чтобы альбом не вышел микстейпом. При этом у него
много конкурентов, которые одновременно с ним бегут на фитах. Необходимо следить
за прогрессом у всех исполнителей и посылать им мотивирующие уведомления. А
именно, ваша программа должна обрабатывать следующие события: RUN user page —
сохранить факт того, что исполнитель под номером user написал page секунд. Если
ранее такой исполнитель не встречался, необходимо его добавить. Гарантируется,
что в рамках одного исполнителя записанные секунды в соответствующих ему
событиях возрастают. CHEER user — сообщить исполнителю user, какая доля
существующих исполнителей (не считая его самого) записала меньшее число секунд,
чем он. Если этот исполнитель на данный момент единственный, доля считается
равной 1. Если для данного исполнителя пока не было ни одного события RUN, доля
считается равной 0, а сам исполнитель не учитывается при вычислении долей для
других до тех пор, пока для него не случится событие RUN.

Формат ввода
В первой строке вводится количество запросов Q — натуральное число, не
превосходящее 10^5. В следующих Q строках в соответствии с описанным выше
форматом вводятся запросы. Гарантируется, что все вводимые числа целые и
положительные, при этом номера исполнителей не превосходят 10^5, а суммарное
время альбома не превосходит 42195 секунд.

Формат вывода
Для каждого запроса CHEER user выведите единственное вещественное число от 0 до
1 — ответ на запрос. Формат вывода этого числа — 6 значащих цифр. То есть
std::setprecision(6), говоря на языке C++.
*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

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
class SegTree {
 public:
  SegTree();
  SegTree(size_t size);
  SegTree(std::vector<Type>& array);
  ~SegTree() {}

  void Build(std::vector<Type>& array);
  Type Get(size_t elem);
  void Update(Segment add_seg, Type add);

  size_t Size() const;

 private:
  size_t size_;
  std::vector<Type> tree_;

  const size_t kCoeffTree = 4;

  void BuildInternal(std::vector<Type>& array, size_t vertex, Segment tree_seg);
  Type GetInternal(size_t vertex, Segment tree_seg, size_t index);
  void UpdateInternal(size_t vertex, Segment tree_seg, Segment add_seg,
                      Type add);
};

template <typename Type>
SegTree<Type>::SegTree() : size_(0) {}

template <typename Type>
SegTree<Type>::SegTree(size_t size) {
  size_ = size;
  tree_.assign(kCoeffTree * size_, 0);
}

template <typename Type>
SegTree<Type>::SegTree(std::vector<Type>& array) {
  build(array);
}

template <typename Type>
void SegTree<Type>::Build(std::vector<Type>& array) {
  size_ = array.size();
  tree_.assign(kCoeffTree * size_, 0);
  build_internal(array, 1, Segment(0, size_));
}

template <typename Type>
Type SegTree<Type>::Get(size_t elem) {
  return GetInternal(1, Segment(0, size_), elem);
}

template <typename Type>
void SegTree<Type>::Update(Segment add_seg, Type add) {
  UpdateInternal(1, Segment(0, size_), add_seg, add);
}

template <typename Type>
size_t SegTree<Type>::Size() const {
  return size_;
}

template <typename Type>
void SegTree<Type>::BuildInternal(std::vector<Type>& array, size_t vertex,
                                  Segment tree_seg) {
  if (tree_seg.left + 1 == tree_seg.right) {
    tree_[vertex] = array[tree_seg.left];
    return;
  }
  BuildInternal(2 * vertex, Segment(tree_seg.left, tree_seg.Middle()));
  BuildInternal(2 * vertex + 1, Segment(tree_seg.Middle(), tree_seg.right));
}

template <typename Type>
Type SegTree<Type>::GetInternal(size_t vertex, Segment tree_seg, size_t index) {
  if (tree_seg.left + 1 == tree_seg.right) {
    return tree_[vertex];
  }

  if (index < tree_seg.Middle()) {
    return tree_[vertex] +
           GetInternal(2 * vertex, Segment(tree_seg.left, tree_seg.Middle()),
                       index);
  }
  return tree_[vertex] + GetInternal(2 * vertex + 1,
                                     Segment(tree_seg.Middle(), tree_seg.right),
                                     index);
}

template <typename Type>
void SegTree<Type>::UpdateInternal(size_t vertex, Segment tree_seg,
                                   Segment add_seg, Type add) {
  if (add_seg.left >= add_seg.right) {
    return;
  }
  if (tree_seg == add_seg) {
    tree_[vertex] += add;
    return;
  }
  UpdateInternal(
      2 * vertex, Segment(tree_seg.left, tree_seg.Middle()),
      Segment(add_seg.left, std::min(add_seg.right, tree_seg.Middle())), add);
  UpdateInternal(
      2 * vertex + 1, Segment(tree_seg.Middle(), tree_seg.right),
      Segment(std::max(add_seg.left, tree_seg.Middle()), add_seg.right), add);
}

void ProcessQueries(std::vector<std::pair<size_t, size_t>>& queries) {
  std::vector<int> composers(100000);
  SegTree<int> segtree(42196);

  size_t count = 0;
  for (const auto& query : queries) {
    if (query.second == 0) {
      if (composers[query.first - 1] == 0) {
        std::cout << 0 << '\n';
      } else if (count == 1) {
        std::cout << 1 << '\n';
      } else {
        std::cout << segtree.Get(composers[query.first - 1] - 1) /
                         static_cast<double>(count - 1)
                  << '\n';
      }
    } else {
      if (composers[query.first - 1] == 0) {
        segtree.Update(Segment(query.second, segtree.Size()), 1);
        composers[query.first - 1] = query.second;
        ++count;
      } else {
        segtree.Update(Segment(composers[query.first - 1], query.second), -1);
        composers[query.first - 1] = query.second;
      }
    }
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(6);

  size_t num_queries;
  std::cin >> num_queries;

  std::vector<std::pair<size_t, size_t>> queries(num_queries);
  for (size_t index = 0; index < num_queries; ++index) {
    std::string command;
    std::cin >> command;
    if (command == "RUN") {
      std::cin >> queries[index];
    } else if (command == "CHEER") {
      std::cin >> queries[index].first;
    }
  }
  ProcessQueries(queries);

  return 0;
}
