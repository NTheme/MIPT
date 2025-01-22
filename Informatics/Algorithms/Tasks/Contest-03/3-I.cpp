/*
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

*/

#include <algorithm>
#include <iostream>
#include <vector>

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

  Segment() : left(0), right(0) {}
  Segment(size_t left_val, size_t right_val)
      : left(left_val), right(right_val) {}

  Segment& operator=(const Segment& other);
  bool operator==(const Segment& other) const;

  size_t Middle() const;
};

Segment& Segment::operator=(const Segment& other) {
  left = other.left;
  right = other.right;
  return *this;
}

bool Segment::operator==(const Segment& other) const {
  return left == other.left && right == other.right;
}

size_t Segment::Middle() const { return (left + right) / 2; }

std::istream& operator>>(std::istream& inp, Segment& segment) {
  inp >> segment.left >> segment.right;
  return inp;
}

template <typename Type>
class SegTree {
 public:
  SegTree();
  SegTree(size_t size);
  ~SegTree() {}

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

void ProcesssQueries(
    const std::vector<Segment>& start,
    const std::vector<std::pair<size_t, std::vector<Segment>>>& queries) {
  const long long kHashMod = 1e9 + 1;
  std::vector<long long> x_coord, y_coord, hash;
  for (const auto& p : queries) {
    for (const auto& q : p.second) {
      x_coord.push_back(q.left);
      y_coord.push_back(q.right);
    }
  }
  x_coord.resize(std::unique(x_coord.begin(), x_coord.end()) - x_coord.begin());
  y_coord.resize(std::unique(y_coord.begin(), y_coord.end()) - y_coord.begin());
  for (const auto& p : start) {
    hash.push_back(p.left * kHashMod + p.right);
  }
  std::sort(x_coord.begin(), x_coord.end());
  std::sort(y_coord.begin(), y_coord.end());
  std::sort(hash.begin(), hash.end());

  SegTree<long long> x_tree(x_coord.size()), y_tree(y_coord.size());
  for (const auto& p : queries) {
    if (p.first == 1) {
      size_t left_x =
          std::lower_bound(x_coord.begin(), x_coord.end(), p.second[0].left) -
          x_coord.begin();
      size_t right_x =
          std::lower_bound(x_coord.begin(), x_coord.end(), p.second[1].left) -
          x_coord.begin();
      size_t left_y =
          std::lower_bound(y_coord.begin(), y_coord.end(), p.second[0].right) -
          y_coord.begin();
      size_t right_y =
          std::lower_bound(y_coord.begin(), y_coord.end(), p.second[1].right) -
          y_coord.begin();
      x_tree.Update({0, left_x + 1}, 1);
      x_tree.Update({right_x + 1, x_tree.Size()}, 1);
      y_tree.Update({left_y + 1, right_y + 1}, 1);
    } else if (p.first == 2) {
      size_t left_x =
          std::lower_bound(x_coord.begin(), x_coord.end(), p.second[0].left) -
          x_coord.begin();
      size_t left_y =
          std::lower_bound(y_coord.begin(), y_coord.end(), p.second[0].right) -
          y_coord.begin();
      long long hash_ptr = p.second[0].left * kHashMod + p.second[0].right;
      size_t start_index =
          std::lower_bound(hash.begin(), hash.end(), hash_ptr) - hash.begin();
      long long result = x_tree.Get(left_x) + y_tree.Get(left_y) +
                         static_cast<long long>(start_index < hash.size() &&
                                                hash[start_index] == hash_ptr);
      result %= 2;
      if (result == 1) {
        std::cout << "YES\n";
      } else {
        std::cout << "NO\n";
      }
    }
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t n, m, k, q;
  std::cin >> n >> m >> k >> q;

  std::vector<Segment> start(k);
  std::cin >> start;
  std::vector<std::pair<size_t, std::vector<Segment>>> queries(q);
  for (size_t index = 0; index < q; ++index) {
    std::cin >> queries[index].first;
    queries[index].second.resize(queries[index].first % 2 + 1);
    std::cin >> queries[index].second;
  }

  ProcesssQueries(start, queries);

  return 0;
}
