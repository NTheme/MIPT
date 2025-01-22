/******************************************
 *  Author : NThemeDEV
 *  Created : Fri Nov 10 2023
 *  File : 12-C.cpp
 ******************************************/

/*
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")
*/

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

static constexpr char NEW_LINE = '\n';

template <typename TypeFirst, typename TypeSecond>
std::istream& operator>>(std::istream& inp,
                         std::pair<TypeFirst, TypeSecond>& pair) {
  inp >> pair.first >> pair.second;
  return inp;
}

template <typename TypeFirst, typename TypeSecond>
std::ostream& operator<<(std::ostream& out,
                         const std::pair<TypeFirst, TypeSecond>& pair) {
  out << pair.first << ' ' << pair.second;
  return out;
}

template <typename Type>
std::istream& operator>>(std::istream& inp, std::vector<Type>& array) {
  for (auto& elem : array) {
    inp >> elem;
  }
  return inp;
}

template <typename Type>
std::ostream& operator<<(std::ostream& out, const std::vector<Type>& array) {
  for (const auto& elem : array) {
    out << elem << ' ';
  }
  return out;
}

struct GCDStruct {
  uint64_t gcd;
  std::pair<int64_t, int64_t> var;
};

GCDStruct GCDI(const uint64_t& first, const uint64_t& second) {
  GCDStruct ret;
  if (first == 0) {
    ret.var = {0, 1};
    ret.gcd = second;
  } else {
    GCDStruct rec = GCDI(second % first, first);
    ret.var = {
        rec.var.second - static_cast<int64_t>(second / first) * rec.var.first,
        rec.var.first};
    ret.gcd = rec.gcd;
  }
  return ret;
}

template <typename TypeFirst, typename TypeSecond>
GCDStruct GCD(const TypeFirst& first, const TypeSecond& second)
  requires std::is_integral_v<TypeFirst> && std::is_integral_v<TypeSecond>
{
  uint64_t first_u = first > 0 ? first : -first;
  uint64_t second_u = second > 0 ? second : -second;
  auto ret = GCDI(first_u, second_u);
  ret.var.first = first > 0 ? ret.var.first : -ret.var.first;
  ret.var.second = second > 0 ? ret.var.second : -ret.var.second;
  return ret;
}

template <typename TypeFirst, typename TypeSecond>
GCDStruct GCD(const std::pair<TypeFirst, TypeSecond>& num) {
  return GCD(num.first, num.second);
}

template <typename TypeFirst, typename TypeSecond>
uint64_t ModularInverse(const TypeFirst& num, const TypeSecond& mod) {
  auto str = GCD(num, mod);
  if (str.gcd != 1) {
    throw std::logic_error("No inversed exists!");
  }

  uint64_t mod_u = mod > 0 ? mod : -mod;
  uint64_t inv_u =
      str.var.first +
      (str.var.first > 0 ? 0 : (-str.var.first + mod_u - 1) / mod_u * mod_u);
  return (inv_u % mod_u + mod_u) % mod_u;
}

struct Segment {
  long long left;
  long long right;

  Segment() = default;
  Segment(const Segment& other) = default;
  Segment(long long left_val, long long right_val);
  ~Segment() = default;

  Segment& operator=(const Segment& other);
  bool operator==(const Segment& other) const;

  long long middle() const;
  size_t length() const;

  Segment leftHalf() const;
  Segment rightHalf() const;
};

Segment::Segment(long long left_val, long long right_val)
    : left(left_val), right(right_val) {}

Segment& Segment::operator=(const Segment& other) {
  left = other.left;
  right = other.right;
  return *this;
}

bool Segment::operator==(const Segment& other) const {
  return left == other.left && right == other.right;
}

long long Segment::middle() const { return (left + right) / 2; }

size_t Segment::length() const { return (right >= left) ? right - left : 0; }

Segment Segment::leftHalf() const { return Segment(left, middle()); }

Segment Segment::rightHalf() const { return Segment(middle(), right); }

std::istream& operator>>(std::istream& inp, Segment& segment) {
  inp >> segment.left >> segment.right;
  return inp;
}

template <typename Type>
class SegmentTree {
 public:
  SegmentTree();
  explicit SegmentTree(size_t size);
  SegmentTree(const std::vector<Type>& array);

  void build(const std::vector<Type>& array);
  Type get(const Segment& query_seg) const;
  size_t size() const;

 private:
  size_t m_size;
  std::vector<Type> m_tree;

  Type merge(const Type& left, const Type& right) const;
  void buildInternal(const std::vector<Type>& array, size_t vertex,
                     const Segment& tree_seg);
  Type getInternal(size_t vertex, const Segment& tree_seg,
                   const Segment& query_seg) const;
};

template <typename Type>
SegmentTree<Type>::SegmentTree() : m_size(0) {}

template <typename Type>
SegmentTree<Type>::SegmentTree(size_t size)
    : m_size(size), m_tree(4 * m_size) {}

template <typename Type>
SegmentTree<Type>::SegmentTree(const std::vector<Type>& array) {
  build(array);
}

template <typename Type>
void SegmentTree<Type>::build(const std::vector<Type>& array) {
  m_size = array.size();
  m_tree.assign(4 * m_size, 0);
  buildInternal(array, 1, Segment(0, m_size));
}

template <typename Type>
Type SegmentTree<Type>::get(const Segment& query_seg) const {
  return getInternal(1, Segment(0, m_size), query_seg);
}


template <typename Type>
size_t SegmentTree<Type>::size() const {
  return m_size;
}

template <typename Type>
Type SegmentTree<Type>::merge(const Type& left, const Type& right) const {
  uint32_t dd = GCD(left, right).gcd;
  return dd;
}

template <typename Type>
void SegmentTree<Type>::buildInternal(const std::vector<Type>& array,
                                      size_t vertex, const Segment& tree_seg) {
  if (tree_seg.length() == 1) {
    m_tree[vertex] = array[tree_seg.left];
    return;
  }

  buildInternal(array, 2 * vertex, tree_seg.leftHalf());
  buildInternal(array, 2 * vertex + 1, tree_seg.rightHalf());
  m_tree[vertex] = merge(m_tree[2 * vertex], m_tree[2 * vertex + 1]);
}

template <typename Type>
Type SegmentTree<Type>::getInternal(size_t vertex, const Segment& tree_seg,
                                    const Segment& query_seg) const {
  if (query_seg.length() == 0) {
    return 0;
  }
  if (tree_seg == query_seg) {
    return m_tree[vertex];
  }

  Segment query_left(query_seg.left,
                     std::min(query_seg.right, tree_seg.middle()));
  Segment query_right(std::max(query_seg.left, tree_seg.middle()),
                      query_seg.right);
  return merge(getInternal(2 * vertex, tree_seg.leftHalf(), query_left),
               getInternal(2 * vertex + 1, tree_seg.rightHalf(), query_right));
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);

  uint64_t size;
  std::cin >> size;
  std::vector<uint32_t> arr(size);
  std::cin >> arr;

  SegmentTree<uint32_t> gcd(arr);

  uint32_t szmin = size + 1;
  for (uint32_t ind = 1; ind <= size; ++ind) {
    uint32_t left = 0, right = ind;
    while (right - left > 1) {
      uint32_t mid = (left + right) / 2;
      if (gcd.get(Segment(mid, ind)) == 1) {
        left = mid;
      } else {
        right = mid;
      }
    }
    if (gcd.get(Segment(left, ind)) == 1) {
      szmin = std::min(szmin, ind - left);
    }
  }

  if (szmin == size + 1) {
    std::cout << -1 << NEW_LINE;
  } else {
    uint32_t num = 0;
    for (uint32_t ind = 0; ind < size; ++ind) {
      num += arr[ind] == 1 ? 1 : 0;
    }
    std::cout << szmin + size - 1 - std::max((uint32_t)1, num) << NEW_LINE;
  }

  std::cout.flush();
  return 0;
}
