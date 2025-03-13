#include <iostream>
#include <memory>

class C;

template <typename T>
class Alloc : std::allocator<T> {
 public:
  void construct(C* p) const;

  using value_type = C;
};

class C {
 private:
  int x = 57;
  C() = default;

  template <typename T>
  friend class Alloc;
};

template <typename T>
void Alloc<T>::construct(C* p) const {
  new (p) C();
}

#include "stackallocator.h"

int main() { List<C, Alloc<C>> lst(5); }
