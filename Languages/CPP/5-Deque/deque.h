#pragma once

#include <compare>
#include <iostream>
#include <vector>

using std::vector;

template <typename Type>
class Deque {
 private:
  template <typename IterType>
  class base_iterator;

 public:
  Deque();
  Deque(const Deque& other);
  Deque(size_t size_new);
  Deque(size_t size_new, const Type& value);
  ~Deque();
  Deque& operator=(Deque other);

  size_t size() const;

  Type& operator[](size_t index);
  const Type& operator[](size_t index) const;

  Type& at(size_t index);
  const Type& at(size_t index) const;

  void push_front(const Type& elem);
  void push_back(const Type& elem);
  void pop_front();
  void pop_back();

  using iterator = base_iterator<Type>;
  using const_iterator = base_iterator<const Type>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;
  const_iterator cbegin() const;
  const_iterator cend() const;

  reverse_iterator rbegin();
  reverse_iterator rend();
  const_reverse_iterator rbegin() const;
  const_reverse_iterator rend() const;
  const_reverse_iterator crbegin() const;
  const_reverse_iterator crend() const;

  void swap(Deque& other);
  void insert(const iterator& place, const Type& elem);
  void erase(const iterator& place);

 private:
  static const ptrdiff_t kBlock = 100;
  vector<Type*> blocks_;
  size_t size_;
  size_t capacity_;
  size_t begin_;

  static void copy(const std::vector<Type*>& blocks_old, std::vector<Type*>& blocks_new,
                   size_t begin, size_t end, size_t place);
  static std::vector<Type*> allocate(size_t capacity_new);
  static void deallocate(std::vector<Type*>& blocks, size_t begin, size_t end);
};

template <typename Type>
template <typename IterType>
class Deque<Type>::base_iterator {
 public:
  using value_type = IterType;
  using difference_type = ptrdiff_t;
  using pointer = IterType*;
  using reference = IterType&;
  using iterator_category = std::random_access_iterator_tag;

  base_iterator();
  base_iterator(IterType* const* block, IterType* elem);
  base_iterator(const base_iterator<Type>& other);
  base_iterator& operator=(const base_iterator& other) = default;
  ~base_iterator() = default;

  difference_type operator-(const base_iterator& other) const;

  bool operator==(const base_iterator& other) const;
  bool operator!=(const base_iterator& other) const;
  std::strong_ordering operator<=>(const base_iterator& other) const;

  base_iterator& operator+=(ptrdiff_t other);
  base_iterator& operator-=(ptrdiff_t other);
  base_iterator operator+(ptrdiff_t other) const;
  base_iterator operator-(ptrdiff_t other) const;

  base_iterator& operator++();
  base_iterator& operator--();
  base_iterator operator++(int);
  base_iterator operator--(int);

  IterType& operator*();
  const IterType& operator*() const;
  IterType* operator->() const;

  IterType* const* GetBlock() const;
  IterType* GetElem() const;
  IterType* GetElemFirst() const;

 private:
  IterType* const* block_;
  IterType* elem_;
  IterType* elem_first_;
  static const ptrdiff_t kBlock = Deque<Type>::kBlock;
};

template <typename Type>
Deque<Type>::Deque() : size_(0), capacity_(0), begin_(0) {
  blocks_ = allocate(1);
  capacity_ = blocks_.size() * kBlock;
  begin_ = capacity_ / 2;
}

template <typename Type>
Deque<Type>::Deque(const Deque<Type>& other) : size_(0), capacity_(0), begin_(0) {
  blocks_ = allocate(other.size_ * 2);
  size_ = other.size_;
  capacity_ = blocks_.size() * kBlock;
  begin_ = (capacity_ - other.size_) / 2;
  copy(other.blocks_, blocks_, other.begin_, other.begin_ + size_, begin_);
}

template <typename Type>
Deque<Type>::Deque(size_t size_new) : Deque(std::max(size_new, 1UL), Type()) {}

template <typename Type>
Deque<Type>::Deque(size_t size_new, const Type& value) : size_(0), capacity_(0), begin_(0) {
  auto blocks_new = allocate(size_new * 2);
  size_t capacity_new = blocks_new.size() * kBlock;
  size_t begin_new = (capacity_new - size_new) / 2;

  Type* block;
  size_t index = begin_new;
  try {
    for (; index < begin_new + size_new; ++index) {
      if (index % kBlock == 0 || index == begin_new) {
        block = blocks_new[index / kBlock];
      }
      new (block + index % kBlock) Type(value);
    }
  } catch (...) {
    deallocate(blocks_new, begin_new, index);
    throw;
  }

  blocks_ = blocks_new;
  size_ = size_new;
  capacity_ = capacity_new;
  begin_ = begin_new;
}

template <typename Type>
Deque<Type>::~Deque() {
  deallocate(blocks_, begin_, begin_ + size_);
}

template <typename Type>
Deque<Type>& Deque<Type>::operator=(Deque other) {
  if (blocks_ != other.blocks_) {
    swap(other);
  }
  return *this;
}

template <typename Type>
size_t Deque<Type>::size() const {
  return size_;
}

template <typename Type>
Type& Deque<Type>::operator[](size_t index) {
  return blocks_[(begin_ + index) / kBlock][(begin_ + index) % kBlock];
}

template <typename Type>
const Type& Deque<Type>::operator[](size_t index) const {
  return blocks_[(begin_ + index) / kBlock][(begin_ + index) % kBlock];
}

template <typename Type>
Type& Deque<Type>::at(size_t index) {
  if (index >= size_) {
    throw std::out_of_range("Deque index is out of range");
  }
  return (*this)[index];
}

template <typename Type>
const Type& Deque<Type>::at(size_t index) const {
  if (index >= size_) {
    throw std::out_of_range("Deque index is out of range");
  }
  return (*this)[index];
}

template <typename Type>
void Deque<Type>::push_front(const Type& elem) {
  if (begin_ == 0) {
    std::vector<Type*> blocks_new;
    try {
      blocks_new = allocate(capacity_ + kBlock);
      std::copy(blocks_.begin(), blocks_.end(), std::back_inserter(blocks_new));
    } catch (...) {
      for (auto& block : blocks_new) {
        delete[] reinterpret_cast<char*>(block);
      }
      return;
    }
    blocks_ = blocks_new;
    begin_ += capacity_ + kBlock;
    capacity_ += capacity_ + kBlock;
  }
  new (blocks_[(begin_ - 1) / kBlock] + (begin_ - 1) % kBlock) Type(elem);
  --begin_;
  ++size_;
}

template <typename Type>
void Deque<Type>::push_back(const Type& elem) {
  if (begin_ + size_ + 1 >= capacity_) {
    std::vector<Type*> blocks_new;
    try {
      blocks_new = allocate(capacity_ + kBlock);
      std::copy(blocks_new.begin(), blocks_new.end(), std::back_inserter(blocks_));
    } catch (...) {
      while (blocks_.size() > capacity_ / kBlock) {
        delete[] reinterpret_cast<char*>(blocks_.back());
      }
      return;
    }
    capacity_ += capacity_ + kBlock;
  }
  new (blocks_[(begin_ + size_) / kBlock] + (begin_ + size_) % kBlock) Type(elem);
  ++size_;
}

template <typename Type>
void Deque<Type>::pop_front() {
  (blocks_[begin_ / kBlock] + begin_ % kBlock)->~Type();
  ++begin_;
  --size_;
}

template <typename Type>
void Deque<Type>::pop_back() {
  (blocks_[(begin_ + size_ - 1) / kBlock] + (begin_ + size_ - 1) % kBlock)->~Type();
  --size_;
}

template <typename Type>
typename Deque<Type>::iterator Deque<Type>::begin() {
  return iterator(blocks_.data() + begin_ / kBlock, blocks_[begin_ / kBlock] + begin_ % kBlock);
}

template <typename Type>
typename Deque<Type>::iterator Deque<Type>::end() {
  return iterator(blocks_.data() + (begin_ + size_) / kBlock,
                  blocks_[(begin_ + size_) / kBlock] + (begin_ + size_) % kBlock);
}

template <typename Type>
typename Deque<Type>::const_iterator Deque<Type>::cbegin() const {
  return const_iterator(blocks_.data() + begin_ / kBlock,
                        blocks_[begin_ / kBlock] + begin_ % kBlock);
}

template <typename Type>
typename Deque<Type>::const_iterator Deque<Type>::cend() const {
  return const_iterator(blocks_.data() + (begin_ + size_) / kBlock,
                        blocks_[(begin_ + size_) / kBlock] + (begin_ + size_) % kBlock);
}

template <typename Type>
typename Deque<Type>::const_iterator Deque<Type>::begin() const {
  return cbegin();
}

template <typename Type>
typename Deque<Type>::const_iterator Deque<Type>::end() const {
  return cend();
}

template <typename Type>
std::reverse_iterator<typename Deque<Type>::iterator> Deque<Type>::rbegin() {
  return std::reverse_iterator<iterator>(end());
}

template <typename Type>
std::reverse_iterator<typename Deque<Type>::iterator> Deque<Type>::rend() {
  return std::reverse_iterator<iterator>(begin());
}

template <typename Type>
std::reverse_iterator<typename Deque<Type>::const_iterator> Deque<Type>::crbegin() const {
  return std::reverse_iterator<const_iterator>(cend());
}

template <typename Type>
std::reverse_iterator<typename Deque<Type>::const_iterator> Deque<Type>::crend() const {
  return std::reverse_iterator<const_iterator>(cbegin());
}

template <typename Type>
std::reverse_iterator<typename Deque<Type>::const_iterator> Deque<Type>::rbegin() const {
  return crbegin();
}

template <typename Type>
std::reverse_iterator<typename Deque<Type>::const_iterator> Deque<Type>::rend() const {
  crend();
}

template <typename Type>
void Deque<Type>::swap(Deque<Type>& other) {
  std::swap(blocks_, other.blocks_);
  std::swap(size_, other.size_);
  std::swap(capacity_, other.capacity_);
  std::swap(begin_, other.begin_);
}

template <typename Type>
void Deque<Type>::insert(const iterator& place, const Type& elem) {
  if (place == begin()) {
    push_front(elem);
    return;
  }
  if (place == end()) {
    push_back(elem);
    return;
  }
  std::vector<Type*> blocks_new;
  try {
    blocks_new = allocate(capacity_ + 1);

    size_t index = place - begin();
    copy(blocks_, blocks_new, begin_, begin_ + index, begin_);
    new (blocks_new[(begin_ + index) / kBlock] + (begin_ + index) % kBlock) Type(elem);
    copy(blocks_, blocks_new, begin_ + index, begin_ + size_, begin_ + index + 1);

  } catch (...) {
    for (auto& block : blocks_new) {
      delete[] reinterpret_cast<char*>(block);
    }
    return;
  }

  deallocate(blocks_, begin_, begin_ + size_);
  blocks_ = blocks_new;
  ++size_;
  ++capacity_;
}

template <typename Type>
void Deque<Type>::erase(const iterator& place) {
  if (place == begin()) {
    pop_front();
    return;
  }
  if (place == end()) {
    pop_back();
    return;
  }

  std::vector<Type*> blocks_new;
  try {
    blocks_new = allocate(capacity_);

    size_t index = place - begin();
    copy(blocks_, blocks_new, begin_, begin_ + index, begin_);
    copy(blocks_, blocks_new, begin_ + index + 1, begin_ + size_, begin_ + index);

  } catch (...) {
    for (auto& block : blocks_new) {
      delete[] reinterpret_cast<char*>(block);
    }
    return;
  }

  deallocate(blocks_, begin_, begin_ + size_);
  blocks_ = blocks_new;
  --size_;
}

template <typename Type>
void Deque<Type>::copy(const std::vector<Type*>& blocks_old, std::vector<Type*>& blocks_new,
                       size_t begin, size_t end, size_t place) {
  Type* block_old;
  Type* block_new;
  for (size_t index = begin; index < end; ++index, ++place) {
    if (index % kBlock == 0 || index == begin) {
      block_old = blocks_old[index / kBlock];
    }
    if (place % kBlock == 0 || index == begin) {
      block_new = blocks_new[place / kBlock];
    }
    new (block_new + place % kBlock) Type(block_old[index % kBlock]);
  }
}

template <typename Type>
std::vector<Type*> Deque<Type>::allocate(size_t capacity_new) {
  std::vector<Type*> blocks_new;
  try {
    for (size_t index = 0; index < (capacity_new + kBlock - 1) / kBlock; ++index) {
      blocks_new.push_back(reinterpret_cast<Type*>(new char[kBlock * sizeof(Type)]));
    }
  } catch (...) {
    for (auto& block : blocks_new) {
      delete[] reinterpret_cast<char*>(block);
    }
    throw;
  }
  return blocks_new;
}

template <typename Type>
void Deque<Type>::deallocate(std::vector<Type*>& blocks, size_t begin, size_t end) {
  Type* block;
  for (size_t index = begin; index < end; ++index) {
    if (index % kBlock == 0 || index == begin) {
      block = blocks[index / kBlock];
    }
    (block + index % kBlock)->~Type();
  }

  for (auto& block : blocks) {
    delete[] reinterpret_cast<char*>(block);
  }
  blocks.resize(0);
}

template <typename Type>
template <typename IterType>
Deque<Type>::base_iterator<IterType>::base_iterator()
    : block_(nullptr), elem_(nullptr), elem_first_(nullptr) {}

template <typename Type>
template <typename IterType>
Deque<Type>::base_iterator<IterType>::base_iterator(const base_iterator<Type>& other)
    : block_(other.GetBlock()), elem_(other.GetElem()), elem_first_(other.GetElemFirst()) {}

template <typename Type>
template <typename IterType>
Deque<Type>::base_iterator<IterType>::base_iterator(IterType* const* block, IterType* elem)
    : block_(block), elem_(elem), elem_first_(*block) {}

template <typename Type>
template <typename IterType>
typename Deque<Type>::template base_iterator<IterType>::difference_type
    Deque<Type>::template base_iterator<IterType>::operator-(
        const base_iterator<IterType>& other) const {
  return (block_ - other.block_) * kBlock + (elem_ - elem_first_) -
         (other.elem_ - other.elem_first_);
}

template <typename Type>
template <typename IterType>
bool Deque<Type>::base_iterator<IterType>::operator==(const base_iterator<IterType>& other) const {
  return elem_ == other.elem_;
}

template <typename Type>
template <typename IterType>
bool Deque<Type>::base_iterator<IterType>::operator!=(const base_iterator& other) const {
  return !(*this == other);
}

template <typename Type>
template <typename IterType>
std::strong_ordering Deque<Type>::base_iterator<IterType>::operator<=>(
    const base_iterator& other) const {
  return *this - other <=> 0;
}

template <typename Type>
template <typename IterType>
typename Deque<Type>::template base_iterator<IterType>&
Deque<Type>::base_iterator<IterType>::operator+=(difference_type other) {
  if (other < 0) {
    *this -= -other;
    return *this;
  }

  difference_type index_new = other + (elem_ - elem_first_);
  if (index_new - kBlock >= 0) {
    block_ += index_new / kBlock;
    elem_first_ = *block_;
  }
  elem_ = elem_first_ + index_new % kBlock;
  return *this;
}

template <typename Type>
template <typename IterType>
typename Deque<Type>::template base_iterator<IterType>&
Deque<Type>::base_iterator<IterType>::operator-=(difference_type other) {
  if (other < 0) {
    *this += -other;
    return *this;
  }

  difference_type index_new = other - (elem_ - elem_first_);
  if (index_new > 0) {
    block_ -= (index_new + kBlock - 1) / kBlock;
    elem_first_ = *block_;
  }
  elem_ = elem_first_ + (kBlock - index_new % kBlock) % kBlock;
  return *this;
}

template <typename Type>
template <typename IterType>
typename Deque<Type>::template base_iterator<IterType>
Deque<Type>::base_iterator<IterType>::operator+(difference_type other) const {
  base_iterator ret = *this;
  ret += other;
  return ret;
}

template <typename Type>
template <typename IterType>
typename Deque<Type>::template base_iterator<IterType>
Deque<Type>::base_iterator<IterType>::operator-(difference_type other) const {
  base_iterator ret = *this;
  ret -= other;
  return ret;
}

template <typename Type>
template <typename IterType>
typename Deque<Type>::template base_iterator<IterType>&
Deque<Type>::base_iterator<IterType>::operator++() {
  return *this += 1;
}

template <typename Type>
template <typename IterType>
typename Deque<Type>::template base_iterator<IterType>&
Deque<Type>::base_iterator<IterType>::operator--() {
  return *this -= 1;
}

template <typename Type>
template <typename IterType>
typename Deque<Type>::template base_iterator<IterType>
Deque<Type>::base_iterator<IterType>::operator++(int) {
  base_iterator old(*this);
  ++*this;
  return old;
}

template <typename Type>
template <typename IterType>
typename Deque<Type>::template base_iterator<IterType>
Deque<Type>::base_iterator<IterType>::operator--(int) {
  base_iterator old(*this);
  --*this;
  return old;
}

template <typename Type>
template <typename IterType>
IterType& Deque<Type>::base_iterator<IterType>::operator*() {
  return *elem_;
}

template <typename Type>
template <typename IterType>
const IterType& Deque<Type>::base_iterator<IterType>::operator*() const {
  return *elem_;
}

template <typename Type>
template <typename IterType>
IterType* Deque<Type>::base_iterator<IterType>::operator->() const {
  return elem_;
}

template <typename Type>
template <typename IterType>
IterType* const* Deque<Type>::base_iterator<IterType>::GetBlock() const {
  return block_;
}

template <typename Type>
template <typename IterType>
IterType* Deque<Type>::base_iterator<IterType>::GetElem() const {
  return elem_;
}

template <typename Type>
template <typename IterType>
IterType* Deque<Type>::base_iterator<IterType>::GetElemFirst() const {
  return elem_first_;
}