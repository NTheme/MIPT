#include <bits/stdc++.h>

struct VerySpecialType {
  int x = 0;
  explicit VerySpecialType(int x) : x(x) {}
  VerySpecialType(const VerySpecialType&) = delete;
  VerySpecialType& operator=(const VerySpecialType&) = delete;

  VerySpecialType(VerySpecialType&&) = default;
  VerySpecialType& operator=(VerySpecialType&&) = default;
};

struct NeitherDefaultNorCopyConstructible {
  VerySpecialType x;

  NeitherDefaultNorCopyConstructible() = delete;
  NeitherDefaultNorCopyConstructible(const NeitherDefaultNorCopyConstructible&) = delete;
  NeitherDefaultNorCopyConstructible& operator=(const NeitherDefaultNorCopyConstructible&) = delete;

  NeitherDefaultNorCopyConstructible(VerySpecialType&& x) : x(std::move(x)) {}
  NeitherDefaultNorCopyConstructible(NeitherDefaultNorCopyConstructible&&) = default;
  NeitherDefaultNorCopyConstructible& operator=(NeitherDefaultNorCopyConstructible&&) = default;

  bool operator==(const NeitherDefaultNorCopyConstructible& other) const {
    return x.x == other.x.x;
  }
};

namespace std {
template <>
struct hash<NeitherDefaultNorCopyConstructible> {
  size_t operator()(const NeitherDefaultNorCopyConstructible& x) const {
    return std::hash<int>()(x.x.x);
  }
};
}  // namespace std

using mytype = NeitherDefaultNorCopyConstructible;
using mypair = std::pair<const mytype, mytype>;

using alloc = std::allocator<mypair>;
using alloc_traits = std::allocator_traits<alloc>;

template <typename... Args>
void fake_emplace(Args&&... args) {
  alloc a;
  mypair* ptr = alloc_traits::allocate(a, 1);
  alloc_traits::construct(a, ptr, std::forward<Args>(args)...);
  std::cout << ptr << '\n';
}

void func(std::pair<mytype, mytype>&& p) {
  // std::cout << p.first.x.x << ' ' << p.second.x.x << '\n';
  fake_emplace(std::move(p));
}

int main() {
  std::pair<NeitherDefaultNorCopyConstructible, NeitherDefaultNorCopyConstructible> p{
      VerySpecialType(1), VerySpecialType(1)};
  func(std::move(p));
  // mypair moved(std::move(p));
}
