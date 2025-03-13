#pragma once

#include <array>
#include <iostream>
#include <memory>

using std::allocator_traits;
using std::array;
using std::byte;

template <size_t Size>
struct StackStorage {
  StackStorage() = default;
  StackStorage(const StackStorage& other) = delete;
  StackStorage& operator=(StackStorage other) = delete;
  ~StackStorage() = default;

  char storage[Size];
  size_t shift = 0;
};

template <typename Type, size_t Size>
class StackAllocator {
 public:
  using Storage = StackStorage<Size>;
  using pointer = Type*;
  using const_pointer = const Type*;
  using void_pointer = void*;
  using const_void_pointer = const void*;
  using value_type = Type;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

  template <typename OtherType>
  struct rebind {
    using other = StackAllocator<OtherType, Size>;
  };

  StackAllocator() = default;
  ~StackAllocator() = default;

  Storage* GetStorage() const;

  StackAllocator(Storage& storage);
  template <typename OtherType>
  StackAllocator(const StackAllocator<OtherType, Size>& other);
  template <typename OtherType>
  StackAllocator& operator=(StackAllocator<OtherType, Size> other);

  template <typename OtherType, size_t SizeOther>
  bool operator==(const StackAllocator<OtherType, SizeOther>& other);

  pointer allocate(size_type size);
  void deallocate(pointer, size_type);

 private:
  Storage* m_storage;
};

template <typename Type, typename Allocator = std::allocator<Type>>
class List {
 private:
  template <typename ValueType>
  class base_iterator;

  struct BaseNode;
  struct Node;

 public:
  using value_type = Type;
  using allocator_type = Allocator;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using reference = Type&;
  using const_reference = const Type&;
  using pointer = Type*;
  using const_pointer = const Type*;
  using iterator = base_iterator<Type>;
  using const_iterator = base_iterator<const Type>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  List() = default;
  List(size_type size);
  List(size_type size, const_reference value);
  List(allocator_type allocator);
  List(size_type size, allocator_type allocator);
  List(size_type size, const_reference value, allocator_type allocator);
  List(const List& other);
  List& operator=(const List& other);
  ~List();

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

  allocator_type get_allocator() const;
  size_type size() const;
  void clear();

  void insert(const_iterator iterator, const_reference value) noexcept;
  void erase(const_iterator iterator);

  void push_front(const_reference value) noexcept;
  void push_back(const_reference value) noexcept;
  void pop_front() noexcept;
  void pop_back() noexcept;

 private:
  using NodeAllocator =
      typename allocator_traits<Allocator>::template rebind_alloc<Node>;
  [[no_unique_address]] NodeAllocator m_allocator;
  BaseNode m_end;
  size_type m_size = 0;

  template <typename... Args>
  void add_node(BaseNode* next, const Args&... args);
  void remove_node(BaseNode* node);
  template <typename... Args>
  void add_nodes(size_type size, BaseNode* next, const Args&... args);
};

template <typename Type, typename Allocator>
struct List<Type, Allocator>::BaseNode {
  BaseNode() = default;
  BaseNode(BaseNode* previous_new, BaseNode* next_new);

  BaseNode* previous = this;
  BaseNode* next = this;
};

template <typename Type, typename Allocator>
struct List<Type, Allocator>::Node : BaseNode {
  Node() = default;
  Node(const_reference value_new);
  Node(Node* previous_new, Node* next_new, const_reference value_new);

  value_type value{};
};

template <typename Type, typename Allocator>
template <typename ValueType>
class List<Type, Allocator>::base_iterator {
 public:
  using value_type = ValueType;
  using difference_type = ptrdiff_t;
  using pointer = ValueType*;
  using reference = ValueType&;
  using const_reference = const ValueType&;
  using iterator_category = std::bidirectional_iterator_tag;

  base_iterator() = default;
  base_iterator(BaseNode* base);
  base_iterator(const base_iterator<Type>& other);
  base_iterator& operator=(const base_iterator& other) = default;
  ~base_iterator() = default;

  template <typename OtherType>
  bool operator==(const base_iterator<OtherType>& other) const;

  base_iterator& operator++();
  base_iterator& operator--();
  base_iterator operator++(int);
  base_iterator operator--(int);

  reference operator*();
  const_reference operator*() const;
  pointer operator->() const;
  BaseNode* base() const;

 private:
  template <typename OtherType>
  friend class base_iterator;

  BaseNode* m_node = nullptr;
};

template <typename Type, size_t Size>
StackStorage<Size>* StackAllocator<Type, Size>::GetStorage() const {
  return m_storage;
}

template <typename Type, size_t Size>
StackAllocator<Type, Size>::StackAllocator(StackStorage<Size>& storage)
    : m_storage(&storage) {}

template <typename Type, size_t Size>
template <typename OtherType>
StackAllocator<Type, Size>::StackAllocator(
    const StackAllocator<OtherType, Size>& other)
    : m_storage(other.GetStorage()) {}

template <typename Type, size_t Size>
template <typename OtherType>
StackAllocator<Type, Size>& StackAllocator<Type, Size>::operator=(
    StackAllocator<OtherType, Size> other) {
  if (m_storage != other.GetStorage()) {
    std::swap(m_storage, other.m_storage);
  }
  return *this;
}

template <typename Type, size_t Size>
template <typename OtherType, size_t SizeOther>
bool StackAllocator<Type, Size>::operator==(
    const StackAllocator<OtherType, SizeOther>& other) {
  return m_storage == other.GetStorage();
}

template <typename Type, size_t Size>
Type* StackAllocator<Type, Size>::allocate(size_t size) {
  auto* ptr = static_cast<void*>(m_storage->storage + m_storage->shift);
  size_t empty = Size - m_storage->shift;
  if (std::align(alignof(Type), 0, ptr, empty) == nullptr) {
    throw std::bad_alloc();
  }
  m_storage->shift = Size - empty + sizeof(Type) * size;
  return reinterpret_cast<Type*>(ptr);
}

template <typename Type, size_t Size>
void StackAllocator<Type, Size>::deallocate(pointer, size_t) {}

template <typename Type, typename Allocator>
List<Type, Allocator>::BaseNode::BaseNode(BaseNode* previous_new,
                                          BaseNode* next_new)
    : previous(previous_new), next(next_new) {}

template <typename Type, typename Allocator>
List<Type, Allocator>::Node::Node(const_reference value_new)
    : BaseNode(), value(value_new) {}

template <typename Type, typename Allocator>
List<Type, Allocator>::Node::Node(Node* previous_new, Node* next_new,
                                  const_reference value_new)
    : BaseNode(previous_new, next_new), value(value_new) {}

template <typename Type, typename Allocator>
List<Type, Allocator>::List(size_type size) {
  add_nodes(size, &m_end);
}

template <typename Type, typename Allocator>
List<Type, Allocator>::List(size_type size, const_reference value) {
  add_nodes(size, &m_end, value);
}

template <typename Type, typename Allocator>
List<Type, Allocator>::List(Allocator allocator) : m_allocator(allocator) {}

template <typename Type, typename Allocator>
List<Type, Allocator>::List(size_type size, Allocator allocator)
    : m_allocator(allocator) {
  add_nodes(size, &m_end);
}

template <typename Type, typename Allocator>
List<Type, Allocator>::List(size_type size, const_reference value,
                            Allocator allocator)
    : m_allocator(allocator) {
  add_nodes(size, &m_end, value);
}

template <typename Type, typename Allocator>
List<Type, Allocator>::List(const List& other) {
  clear();
  m_allocator =
      allocator_traits<NodeAllocator>::select_on_container_copy_construction(
          other.m_allocator);
  try {
    for (const auto& element : other) {
      add_node(&m_end, element);
    }
  } catch (std::bad_alloc& exc) {
    clear();
    throw exc;
  }
}

template <typename Type, typename Allocator>
List<Type, Allocator>& List<Type, Allocator>::operator=(const List& other) {
  if(end() == other.end()) {
    return *this;
  }
  if (allocator_traits<
          NodeAllocator>::propagate_on_container_copy_assignment::value) {
    m_allocator = other.m_allocator;
  }
  size_t old_size = m_size;
  try {
    for (const auto& element : other) {
      add_node(&m_end, element);
    }
    while (m_size > other.m_size) {
      remove_node(m_end.next);
    }
  } catch (std::bad_alloc& exc) {
    while (m_size > old_size) {
      remove_node(m_end.previous);
    }
    throw exc;
  }
  return *this;
}

template <typename Type, typename Allocator>
List<Type, Allocator>::~List() {
  clear();
}

template <typename Type, typename Allocator>
typename List<Type, Allocator>::iterator List<Type, Allocator>::begin() {
  return iterator(m_end.next);
}

template <typename Type, typename Allocator>
typename List<Type, Allocator>::iterator List<Type, Allocator>::end() {
  return iterator(&m_end);
}

template <typename Type, typename Allocator>
typename List<Type, Allocator>::const_iterator List<Type, Allocator>::cbegin()
    const {
  return const_iterator(m_end.next);
}

template <typename Type, typename Allocator>
typename List<Type, Allocator>::const_iterator List<Type, Allocator>::cend()
    const {
  return const_iterator(const_cast<BaseNode*>(&m_end));
}

template <typename Type, typename Allocator>
typename List<Type, Allocator>::const_iterator List<Type, Allocator>::begin()
    const {
  return cbegin();
}

template <typename Type, typename Allocator>
typename List<Type, Allocator>::const_iterator List<Type, Allocator>::end()
    const {
  return cend();
}

template <typename Type, typename Allocator>
std::reverse_iterator<typename List<Type, Allocator>::iterator>
List<Type, Allocator>::rbegin() {
  return std::reverse_iterator<iterator>(end());
}

template <typename Type, typename Allocator>
std::reverse_iterator<typename List<Type, Allocator>::iterator>
List<Type, Allocator>::rend() {
  return std::reverse_iterator<iterator>(begin());
}

template <typename Type, typename Allocator>
std::reverse_iterator<typename List<Type, Allocator>::const_iterator>
List<Type, Allocator>::crbegin() const {
  return std::reverse_iterator<const_iterator>(cend());
}

template <typename Type, typename Allocator>
std::reverse_iterator<typename List<Type, Allocator>::const_iterator>
List<Type, Allocator>::crend() const {
  return std::reverse_iterator<const_iterator>(cbegin());
}

template <typename Type, typename Allocator>
std::reverse_iterator<typename List<Type, Allocator>::const_iterator>
List<Type, Allocator>::rbegin() const {
  return crbegin();
}

template <typename Type, typename Allocator>
std::reverse_iterator<typename List<Type, Allocator>::const_iterator>
List<Type, Allocator>::rend() const {
  return crend();
}

template <typename Type, typename Allocator>
Allocator List<Type, Allocator>::get_allocator() const {
  return m_allocator;
}

template <typename Type, typename Allocator>
size_t List<Type, Allocator>::size() const {
  return m_size;
}

template <typename Type, typename Allocator>
void List<Type, Allocator>::clear() {
  while (m_end.previous != &m_end) {
    remove_node(m_end.previous);
  }
}

template <typename Type, typename Allocator>
void List<Type, Allocator>::insert(const_iterator iterator,
                                   const_reference value) noexcept {
  add_nodes(1, iterator.base(), value);
}

template <typename Type, typename Allocator>
void List<Type, Allocator>::erase(const_iterator iterator) {
  remove_node(iterator.base());
}

template <typename Type, typename Allocator>
void List<Type, Allocator>::push_front(const_reference value) noexcept {
  add_nodes(1, m_end.next, value);
}

template <typename Type, typename Allocator>
void List<Type, Allocator>::push_back(const_reference value) noexcept {
  add_nodes(1, &m_end, value);
}

template <typename Type, typename Allocator>
void List<Type, Allocator>::pop_front() noexcept {
  remove_node(m_end.next);
}

template <typename Type, typename Allocator>
void List<Type, Allocator>::pop_back() noexcept {
  remove_node(m_end.previous);
}

template <typename Type, typename Allocator>
template <typename... Args>
void List<Type, Allocator>::add_node(BaseNode* next, const Args&... args) {
  Node* node = nullptr;
  try {
    node = allocator_traits<NodeAllocator>::allocate(m_allocator, 1);
    allocator_traits<NodeAllocator>::construct(m_allocator, node, args...);
    node->previous = next->previous;
    node->next = next;
    node->next->previous = node->previous->next = node;
    ++m_size;
  } catch (...) {
    throw std::bad_alloc();
  }
}

template <typename Type, typename Allocator>
void List<Type, Allocator>::remove_node(BaseNode* node) {
  node->previous->next = node->next;
  node->next->previous = node->previous;
  allocator_traits<NodeAllocator>::destroy(m_allocator,
                                           static_cast<Node*>(node));
  allocator_traits<NodeAllocator>::deallocate(m_allocator,
                                              static_cast<Node*>(node), 1);
  --m_size;
}

template <typename Type, typename Allocator>
template <typename... Args>
void List<Type, Allocator>::add_nodes(size_type size, BaseNode* next,
                                      const Args&... args) {
  size_type size_last = m_size;
  try {
    for (size_type index = 0; index < size; ++index) {
      add_node(next, args...);
    }
  } catch (std::bad_alloc& exc) {
    while (m_size != size_last) {
      remove_node(next->previous);
    }
    throw exc;
  }
}

template <typename Type, typename Allocator>
template <typename ValueType>
List<Type, Allocator>::base_iterator<ValueType>::base_iterator(BaseNode* base)
    : m_node(base) {}

template <typename Type, typename Allocator>
template <typename ValueType>
List<Type, Allocator>::base_iterator<ValueType>::base_iterator(
    const base_iterator<Type>& other)
    : m_node(other.base()) {}

template <typename Type, typename Allocator>
template <typename ValueType>
template <typename OtherType>
bool List<Type, Allocator>::base_iterator<ValueType>::operator==(
    const base_iterator<OtherType>& other) const {
  return m_node == other.m_node;
}

template <typename Type, typename Allocator>
template <typename ValueType>
typename List<Type, Allocator>::template base_iterator<ValueType>&
List<Type, Allocator>::base_iterator<ValueType>::operator++() {
  m_node = m_node->next;
  return *this;
}

template <typename Type, typename Allocator>
template <typename ValueType>
typename List<Type, Allocator>::template base_iterator<ValueType>&
List<Type, Allocator>::base_iterator<ValueType>::operator--() {
  m_node = m_node->previous;
  return *this;
}

template <typename Type, typename Allocator>
template <typename ValueType>
typename List<Type, Allocator>::template base_iterator<ValueType>
List<Type, Allocator>::base_iterator<ValueType>::operator++(int) {
  base_iterator old(*this);
  m_node = m_node->next;
  return old;
}

template <typename Type, typename Allocator>
template <typename ValueType>
typename List<Type, Allocator>::template base_iterator<ValueType>
List<Type, Allocator>::base_iterator<ValueType>::operator--(int) {
  base_iterator old(*this);
  m_node = m_node->previous;
  return old;
}

template <typename Type, typename Allocator>
template <typename ValueType>
ValueType& List<Type, Allocator>::base_iterator<ValueType>::operator*() {
  return static_cast<Node*>(m_node)->value;
}

template <typename Type, typename Allocator>
template <typename ValueType>
const ValueType& List<Type, Allocator>::base_iterator<ValueType>::operator*()
    const {
  return static_cast<Node*>(m_node)->value;
}

template <typename Type, typename Allocator>
template <typename ValueType>
ValueType* List<Type, Allocator>::base_iterator<ValueType>::operator->() const {
  return &static_cast<Node*>(m_node)->value;
}

template <typename Type, typename Allocator>
template <typename ValueType>
typename List<Type, Allocator>::BaseNode*
List<Type, Allocator>::base_iterator<ValueType>::base() const {
  return m_node;
}
