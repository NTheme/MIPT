#pragma once

#include <iostream>
#include <memory>
#include <vector>

template <typename Key, typename Value, typename Hash = std::hash<Key>,
          typename Equal = std::equal_to<Key>,
          typename Allocator = std::allocator<std::pair<const Key, Value>>>
class UnorderedMap {
 private:
  struct BaseNode;
  struct Node;

  template <typename IteratorType>
  class base_iterator;

 public:
  using key_type = Key;
  using mapped_type = Value;
  using value_type = std::pair<const key_type, mapped_type>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using hasher = Hash;
  using key_equal = Equal;
  using allocator_type = Allocator;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = typename std::allocator_traits<Allocator>::pointer;
  using const_pointer = typename std::allocator_traits<Allocator>::const_pointer;
  using iterator = base_iterator<value_type>;
  using const_iterator = base_iterator<const value_type>;
  using node_type = Node;
  using insert_return_type = std::pair<iterator, bool>;

  using NodeType = value_type;

  UnorderedMap() noexcept = default;
  ~UnorderedMap() noexcept;

  UnorderedMap(const UnorderedMap& other);
  UnorderedMap(UnorderedMap&& other);
  UnorderedMap& operator=(const UnorderedMap& other);
  UnorderedMap& operator=(UnorderedMap&& other);

  iterator begin() noexcept;
  iterator end() noexcept;
  const_iterator begin() const noexcept;
  const_iterator end() const noexcept;
  const_iterator cbegin() const noexcept;
  const_iterator cend() const noexcept;

  bool empty() const noexcept;
  size_type size() const noexcept;
  void clear() noexcept;

  insert_return_type insert(const std::pair<Key, Value>& value);
  insert_return_type insert(std::pair<Key, Value>&& value);
  template <typename InputIt>
  void insert(InputIt first, InputIt last);

  template <typename... Args>
  insert_return_type emplace(Args&&... args) noexcept;

  iterator erase(iterator pos) noexcept;
  iterator erase(iterator first, iterator last) noexcept;

  void swap(UnorderedMap& other) noexcept;

  iterator find(const key_type& key) noexcept;
  const_iterator find(const key_type& key) const noexcept;

  mapped_type& operator[](const key_type& key) noexcept;
  mapped_type& operator[](key_type&& key) noexcept;

  mapped_type& at(const key_type& key);
  const mapped_type& at(const key_type& key) const;

  double load_factor() const noexcept;
  double max_load_factor() const noexcept;
  void max_load_factor(double max_load) noexcept;
  void rehash(size_type size) noexcept;
  void reserve(size_type size) noexcept;

 private:
  using NodeAllocator = typename std::allocator_traits<Allocator>::template rebind_alloc<Node>;

  double m_max_load = 0.5;
  size_type m_size = 0;
  BaseNode m_end;
  std::vector<BaseNode*> m_buckets;
  [[no_unique_address]] hasher m_hasher;
  [[no_unique_address]] key_equal m_equal;
  [[no_unique_address]] NodeAllocator m_allocator;

  void place(node_type* node) noexcept;
  BaseNode* exist(const key_type& key) const noexcept;
};

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
struct UnorderedMap<Key, Value, Hash, Equal, Allocator>::BaseNode {
  BaseNode* previous = this;
  BaseNode* cur = this;
  BaseNode* next = this;

  BaseNode() = default;
  BaseNode(BaseNode* previous_new, BaseNode* next_new);
};

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
struct UnorderedMap<Key, Value, Hash, Equal, Allocator>::Node : BaseNode {
  value_type value;
  size_type hash;
};

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
template <typename IteratorType>
class UnorderedMap<Key, Value, Hash, Equal, Allocator>::base_iterator {
 public:
  using value_type = IteratorType;
  using difference_type = ptrdiff_t;
  using pointer = IteratorType*;
  using reference = IteratorType&;
  using const_reference = const IteratorType&;
  using iterator_category = std::bidirectional_iterator_tag;

  base_iterator() noexcept = default;
  ~base_iterator() noexcept = default;

  explicit base_iterator(BaseNode* base) noexcept;
  base_iterator(const base_iterator<UnorderedMap::value_type>& other) noexcept;
  base_iterator& operator=(const base_iterator& other) noexcept = default;

  bool operator==(const base_iterator& other) const noexcept;

  base_iterator& operator++();
  base_iterator& operator--();
  base_iterator operator++(int);
  base_iterator operator--(int);

  reference operator*();
  const_reference operator*() const;
  pointer operator->();

  BaseNode* base() const noexcept;

 private:
  BaseNode* m_node = nullptr;
};

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
UnorderedMap<Key, Value, Hash, Equal, Allocator>::BaseNode::BaseNode(BaseNode* previous_new,
                                                                     BaseNode* next_new)
    : previous(previous_new), next(next_new) {}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
template <typename IteratorType>
UnorderedMap<Key, Value, Hash, Equal, Allocator>::base_iterator<IteratorType>::base_iterator(
    BaseNode* base) noexcept
    : m_node(base) {}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
template <typename IteratorType>
UnorderedMap<Key, Value, Hash, Equal, Allocator>::base_iterator<IteratorType>::base_iterator(
    const base_iterator<UnorderedMap::value_type>& other) noexcept
    : m_node(other.base()) {}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
template <typename IteratorType>
bool UnorderedMap<Key, Value, Hash, Equal, Allocator>::base_iterator<IteratorType>::operator==(
    const base_iterator& other) const noexcept {
  return m_node == other.m_node;
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
template <typename IteratorType>
typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::template base_iterator<IteratorType>&
UnorderedMap<Key, Value, Hash, Equal, Allocator>::base_iterator<IteratorType>::operator++() {
  m_node = m_node->next;
  return *this;
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
template <typename IteratorType>
typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::template base_iterator<IteratorType>&
UnorderedMap<Key, Value, Hash, Equal, Allocator>::base_iterator<IteratorType>::operator--() {
  m_node = m_node->previous;
  return *this;
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
template <typename IteratorType>
typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::template base_iterator<IteratorType>
UnorderedMap<Key, Value, Hash, Equal, Allocator>::base_iterator<IteratorType>::operator++(int) {
  base_iterator old(*this);
  m_node = m_node->next;
  return old;
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
template <typename IteratorType>
typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::template base_iterator<IteratorType>
UnorderedMap<Key, Value, Hash, Equal, Allocator>::base_iterator<IteratorType>::operator--(int) {
  base_iterator old(*this);
  m_node = m_node->previous;
  return old;
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
template <typename IteratorType>
typename UnorderedMap<Key, Value, Hash, Equal,
                      Allocator>::template base_iterator<IteratorType>::reference
UnorderedMap<Key, Value, Hash, Equal, Allocator>::base_iterator<IteratorType>::operator*() {
  return static_cast<Node*>(m_node)->value;
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
template <typename IteratorType>
typename UnorderedMap<Key, Value, Hash, Equal,
                      Allocator>::template base_iterator<IteratorType>::const_reference
UnorderedMap<Key, Value, Hash, Equal, Allocator>::base_iterator<IteratorType>::operator*() const {
  return static_cast<Node*>(m_node)->value;
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
template <typename IteratorType>
typename UnorderedMap<Key, Value, Hash, Equal,
                      Allocator>::template base_iterator<IteratorType>::pointer
UnorderedMap<Key, Value, Hash, Equal, Allocator>::base_iterator<IteratorType>::operator->() {
  return &static_cast<Node*>(m_node)->value;
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
template <typename IteratorType>
typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::BaseNode* UnorderedMap<
    Key, Value, Hash, Equal, Allocator>::base_iterator<IteratorType>::base() const noexcept {
  return m_node;
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
UnorderedMap<Key, Value, Hash, Equal, Allocator>::~UnorderedMap() noexcept {
  clear();
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
UnorderedMap<Key, Value, Hash, Equal, Allocator>::UnorderedMap(const UnorderedMap& other) {
  m_allocator = std::allocator_traits<NodeAllocator>::select_on_container_copy_construction(
      other.m_allocator);
  for (auto& item : other) {
    insert(item);
  }
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
UnorderedMap<Key, Value, Hash, Equal, Allocator>::UnorderedMap(UnorderedMap&& other) {
  m_allocator = std::allocator_traits<NodeAllocator>::select_on_container_copy_construction(
      other.m_allocator);
  swap(other);
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
UnorderedMap<Key, Value, Hash, Equal, Allocator>&
UnorderedMap<Key, Value, Hash, Equal, Allocator>::operator=(const UnorderedMap& other) {
  if (std::allocator_traits<NodeAllocator>::propagate_on_container_copy_assignment::value) {
    m_allocator = other.m_allocator;
  }
  UnorderedMap copy = other;
  swap(copy);
  return *this;
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
UnorderedMap<Key, Value, Hash, Equal, Allocator>&
UnorderedMap<Key, Value, Hash, Equal, Allocator>::operator=(UnorderedMap&& other) {
  if (std::allocator_traits<NodeAllocator>::propagate_on_container_copy_assignment::value) {
    m_allocator = other.m_allocator;
  }
  UnorderedMap copy = std::move(other);
  swap(copy);
  return *this;
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::iterator
UnorderedMap<Key, Value, Hash, Equal, Allocator>::begin() noexcept {
  return iterator(m_end.next);
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::iterator
UnorderedMap<Key, Value, Hash, Equal, Allocator>::end() noexcept {
  return iterator(&m_end);
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::const_iterator
UnorderedMap<Key, Value, Hash, Equal, Allocator>::cbegin() const noexcept {
  return const_iterator(const_cast<BaseNode*>(m_end.next));
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::const_iterator
UnorderedMap<Key, Value, Hash, Equal, Allocator>::cend() const noexcept {
  return const_iterator(const_cast<BaseNode*>(&m_end));
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::const_iterator
UnorderedMap<Key, Value, Hash, Equal, Allocator>::begin() const noexcept {
  return cbegin();
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::const_iterator
UnorderedMap<Key, Value, Hash, Equal, Allocator>::end() const noexcept {
  return cend();
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
bool UnorderedMap<Key, Value, Hash, Equal, Allocator>::empty() const noexcept {
  return m_size == 0;
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::size_type
UnorderedMap<Key, Value, Hash, Equal, Allocator>::size() const noexcept {
  return m_size;
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
void UnorderedMap<Key, Value, Hash, Equal, Allocator>::clear() noexcept {
  while (m_size > 0) {
    erase(begin());
  }
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::insert_return_type
UnorderedMap<Key, Value, Hash, Equal, Allocator>::insert(const std::pair<Key, Value>& value) {
  return emplace(value);
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::insert_return_type
UnorderedMap<Key, Value, Hash, Equal, Allocator>::insert(std::pair<Key, Value>&& value) {
  return emplace(std::move(value));
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
template <typename InputIt>
void UnorderedMap<Key, Value, Hash, Equal, Allocator>::insert(InputIt first, InputIt last) {
  for (auto it = first; it != last; ++it) {
    insert(*it);
  }
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
template <typename... Args>
typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::insert_return_type
UnorderedMap<Key, Value, Hash, Equal, Allocator>::emplace(Args&&... args) noexcept {
  reserve(m_size + 1);
  node_type* node = nullptr;
  try {
    node = std::allocator_traits<NodeAllocator>::allocate(m_allocator, 1);
    std::allocator_traits<NodeAllocator>::construct(m_allocator, &node->value,
                                                    std::forward<Args>(args)...);
    node->hash = m_hasher(node->value.first);
    node->cur = node;

    auto* inside = exist(node->value.first);
    if (inside != nullptr) {
      std::allocator_traits<NodeAllocator>::destroy(m_allocator, &node->value);
      std::allocator_traits<NodeAllocator>::deallocate(m_allocator, node, 1);
      return insert_return_type(iterator(inside), false);
    }
    place(node);
    ++m_size;
    return insert_return_type(iterator(node), true);
  } catch (...) {
    std::allocator_traits<NodeAllocator>::destroy(m_allocator, &node->value);
    std::allocator_traits<NodeAllocator>::deallocate(m_allocator, node, 1);
    return insert_return_type(end(), false);
  }
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::iterator
UnorderedMap<Key, Value, Hash, Equal, Allocator>::erase(iterator pos) noexcept {
  auto next = pos.base()->next;
  auto hash_mod = static_cast<Node*>(pos.base())->hash % m_buckets.size();
  auto*& bucket = m_buckets[hash_mod];
  if (bucket == pos.base()) {
    if (next != &m_end && static_cast<Node*>(next)->hash % m_buckets.size() == hash_mod) {
      bucket = next;
    } else {
      bucket = nullptr;
    }
  }

  pos.base()->previous->next = pos.base()->next;
  pos.base()->next->previous = pos.base()->previous;
  std::allocator_traits<NodeAllocator>::destroy(m_allocator,
                                                &static_cast<Node*>(pos.base())->value);
  std::allocator_traits<NodeAllocator>::deallocate(m_allocator, static_cast<Node*>(pos.base()), 1);
  --m_size;
  return iterator(next);
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::iterator
UnorderedMap<Key, Value, Hash, Equal, Allocator>::erase(iterator first, iterator last) noexcept {
  while (first != last) {
    auto next = iterator(first.base()->next);
    erase(first);
    first = next;
  }
  return last;
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
void UnorderedMap<Key, Value, Hash, Equal, Allocator>::swap(UnorderedMap& other) noexcept {
  std::swap(m_end.previous, other.m_end.previous);
  std::swap(m_end.next, other.m_end.next);
  if (m_size > 0 && other.m_size > 0) {
    std::swap(m_end.previous->next, other.m_end.previous->next);
    std::swap(m_end.next->previous, other.m_end.next->previous);
  } else if (other.m_size == 0) {
    std::swap(m_end.previous, other.m_end.previous->next);
    std::swap(m_end.next, other.m_end.next->previous);
  } else if (m_size == 0) {
    std::swap(other.m_end.previous, m_end.previous->next);
    std::swap(other.m_end.next, m_end.next->previous);
  } else {
    std::swap(m_end.previous, other.m_end.previous);
    std::swap(m_end.next, other.m_end.next);
  }
  std::swap(m_size, other.m_size);
  std::swap(m_buckets, other.m_buckets);
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::iterator
UnorderedMap<Key, Value, Hash, Equal, Allocator>::find(const key_type& key) noexcept {
  auto* inside = exist(key);
  return inside == nullptr ? end() : iterator(inside);
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::mapped_type&
UnorderedMap<Key, Value, Hash, Equal, Allocator>::operator[](const key_type& key) noexcept {
  auto* inside = exist(key);
  if (inside == nullptr) {
    return emplace(key, std::move(Value())).first->second;
  }
  return static_cast<Node*>(inside)->value.second;
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::mapped_type&
UnorderedMap<Key, Value, Hash, Equal, Allocator>::operator[](key_type&& key) noexcept {
  auto* inside = exist(key);
  if (inside == nullptr) {
    return emplace(std::move(key), std::move(Value())).first->second;
  }
  return static_cast<Node*>(inside)->value.second;
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::mapped_type&
UnorderedMap<Key, Value, Hash, Equal, Allocator>::at(const key_type& key) {
  auto* inside = exist(key);
  if (inside == nullptr) {
    throw std::runtime_error("Key doesn't exist!");
  }
  return static_cast<Node*>(inside)->value.second;
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
const typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::mapped_type&
UnorderedMap<Key, Value, Hash, Equal, Allocator>::at(const key_type& key) const {
  return at(key);
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
double UnorderedMap<Key, Value, Hash, Equal, Allocator>::load_factor() const noexcept {
  return m_size / static_cast<double>(m_buckets.size());
}
template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
double UnorderedMap<Key, Value, Hash, Equal, Allocator>::max_load_factor() const noexcept {
  return m_max_load;
}
template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
void UnorderedMap<Key, Value, Hash, Equal, Allocator>::max_load_factor(double max_load) noexcept {
  m_max_load = max_load;
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
void UnorderedMap<Key, Value, Hash, Equal, Allocator>::rehash(size_type size) noexcept {
  auto* cur = m_end.next;
  m_buckets.assign(size, nullptr);
  m_end.previous = m_end.next = &m_end;
  for (size_t hash_modex = 0; hash_modex < m_size; ++hash_modex) {
    auto* next = cur->next;
    place(static_cast<Node*>(cur));
    cur = next;
  }
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
void UnorderedMap<Key, Value, Hash, Equal, Allocator>::reserve(size_type size) noexcept {
  if (size < m_max_load * m_buckets.size()) {
    return;
  }
  rehash(2 * size / m_max_load);
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
void UnorderedMap<Key, Value, Hash, Equal, Allocator>::place(node_type* node) noexcept {
  auto*& bucket = m_buckets[node->hash % m_buckets.size()];
  node->previous = (bucket == nullptr) ? m_end.previous : bucket->previous;
  node->next = (bucket == nullptr) ? &m_end : bucket;
  node->next->previous = node->previous->next = node;
  bucket = node;
}

template <typename Key, typename Value, typename Hash, typename Equal, typename Allocator>
typename UnorderedMap<Key, Value, Hash, Equal, Allocator>::BaseNode*
UnorderedMap<Key, Value, Hash, Equal, Allocator>::exist(const key_type& key) const noexcept {
  size_type hash = m_hasher(key), hash_mod = 0;
  if (m_buckets.size() == 0 || m_buckets[hash_mod = hash % m_buckets.size()] == nullptr) {
    return nullptr;
  }
  for (auto* item = m_buckets[hash_mod];
       item != &m_end && static_cast<Node*>(item)->hash % m_buckets.size() == hash_mod;
       item = item->next) {
    if (m_equal(static_cast<Node*>(item)->value.first, key)) {
      return item;
    }
  }
  return nullptr;
}
