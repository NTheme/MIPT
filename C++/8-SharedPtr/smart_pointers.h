#pragma once

#include <cstddef>
#include <memory>

template <typename Type>
class SharedPtr;

template <typename Type>
class WeakPtr;

class ControlBlockBase {
  template <typename Type>
  friend class SharedPtr;

  template <typename Type>
  friend class WeakPtr;

  template <typename OtherType, typename Allocator, typename... Args>
  friend SharedPtr<OtherType> allocateShared(const Allocator& alloc, Args&&... args);

  size_t shared_cnt = 0;
  size_t weak_cnt = 0;

  virtual ~ControlBlockBase() noexcept = default;
  virtual void delete_obj() noexcept = 0;
  virtual void delete_cls() noexcept = 0;
};

template <typename Type>
class SharedPtr {
 public:
  using element_type = Type;
  using weak_type = WeakPtr<element_type>;

  SharedPtr() = default;
  ~SharedPtr() noexcept;

  template <typename OtherType = Type, typename Deleter = std::default_delete<Type>,
            typename Allocator = std::allocator<Type>>
  SharedPtr(OtherType* ptr, Deleter deleter = Deleter(), Allocator allocator = Allocator())
    requires std::is_convertible_v<OtherType*, Type*>;

  SharedPtr(const SharedPtr& other);
  template <typename OtherType = Type>
  SharedPtr(const SharedPtr<OtherType>& other)
    requires std::is_convertible_v<OtherType*, Type*>;

  SharedPtr(SharedPtr&& other);
  template <typename OtherType = Type>
  SharedPtr(SharedPtr<OtherType>&& other)
    requires std::is_convertible_v<OtherType*, Type*>;

  SharedPtr& operator=(const SharedPtr& other);
  template <typename OtherType = Type>
  SharedPtr& operator=(const SharedPtr<OtherType>& other)
    requires std::is_convertible_v<OtherType*, Type*>;

  SharedPtr& operator=(SharedPtr&& other);
  template <typename OtherType>
  SharedPtr& operator=(SharedPtr<OtherType>&& other)
    requires std::is_convertible_v<OtherType*, Type*>;

  template <typename OtherType = Type, typename Deleter = std::default_delete<Type>,
            typename Allocator = std::allocator<Type>>
  void reset(OtherType* ptr = nullptr, Deleter deleter = Deleter(),
             Allocator allocator = Allocator()) noexcept
    requires std::is_convertible_v<OtherType*, Type*>;

  void swap(SharedPtr& other) noexcept;
  size_t use_count() const noexcept;

  element_type* get() const noexcept;
  element_type* operator->() const noexcept;
  element_type& operator*() const noexcept;

 private:
  template <typename OtherType>
  friend class SharedPtr;

  template <typename OtherType>
  friend class WeakPtr;

  template <typename OtherType>
  friend class EnableSharedFromThis;

  template <typename OtherType, typename Allocator, typename... Args>
  friend SharedPtr<OtherType> allocateShared(const Allocator& alloc, Args&&... args);

  template <typename Deleter, typename Allocator>
  struct ControlBlockRegular;

  template <typename Allocator>
  struct ControlBlockShared;

  element_type* m_ptr = nullptr;
  ControlBlockBase* m_block = nullptr;

  template <typename OtherType = Type>
  explicit SharedPtr(const WeakPtr<OtherType>& other)
    requires std::is_convertible_v<OtherType*, Type*>;
};

template <typename Type>
template <typename Deleter, typename Allocator>
struct SharedPtr<Type>::ControlBlockRegular : public ControlBlockBase {
  Type* ptr = nullptr;
  [[no_unique_address]] Deleter deleter = Deleter();
  [[no_unique_address]] Allocator allocator = Allocator();

  ControlBlockRegular(Deleter n_deleter, Allocator n_allocator) noexcept;
  ~ControlBlockRegular() noexcept = default;
  void delete_obj() noexcept final;
  void delete_cls() noexcept final;
};

template <typename Type>
template <typename Allocator>
struct SharedPtr<Type>::ControlBlockShared : public ControlBlockBase {
  element_type obj;
  [[no_unique_address]] Allocator allocator = Allocator();

  template <typename... Args>
  ControlBlockShared(Allocator n_allocator, Args&&... args) noexcept;
  ~ControlBlockShared() noexcept = default;
  void delete_obj() noexcept final;
  void delete_cls() noexcept final;
};

template <typename Type>
class WeakPtr {
 public:
  using element_type = Type;

  WeakPtr() noexcept = default;
  ~WeakPtr() noexcept;

  WeakPtr(const WeakPtr& other);
  template <typename OtherType = Type>
  WeakPtr(const WeakPtr<OtherType>& other)
    requires std::is_convertible_v<OtherType*, Type*>;

  WeakPtr(WeakPtr&& other);
  template <typename OtherType = Type>
  WeakPtr(WeakPtr<OtherType>&& other)
    requires std::is_convertible_v<OtherType*, Type*>;

  template <typename OtherType = Type>
  WeakPtr(const SharedPtr<OtherType>& other)
    requires std::is_convertible_v<OtherType*, Type*>;

  WeakPtr& operator=(const WeakPtr& other);
  template <typename OtherType = Type>
  WeakPtr& operator=(const WeakPtr<OtherType>& other)
    requires std::is_convertible_v<OtherType*, Type*>;

  WeakPtr& operator=(WeakPtr&& other);
  template <typename OtherType = Type>
  WeakPtr& operator=(WeakPtr<OtherType>&& other)
    requires std::is_convertible_v<OtherType*, Type*>;

  template <typename OtherType = Type>
  WeakPtr& operator=(const SharedPtr<OtherType>& other)
    requires std::is_convertible_v<OtherType*, Type*>;

  void swap(WeakPtr& other) noexcept;
  bool expired() const noexcept;
  size_t use_count() const noexcept;
  SharedPtr<element_type> lock() const noexcept;

 private:
  template <typename OtherType>
  friend class WeakPtr;

  template <typename OtherType>
  friend class SharedPtr;

  element_type* m_ptr = nullptr;
  ControlBlockBase* m_block = nullptr;
};

template <typename Type>
class EnableSharedFromThis {
 public:
  EnableSharedFromThis() noexcept = default;

  SharedPtr<Type> shared_from_this();
  SharedPtr<const Type> shared_from_this() const;

  WeakPtr<Type> weak_from_this();
  WeakPtr<const Type> weak_from_this() const;

 private:
  template <typename OtherType>
  friend class SharedPtr;

  WeakPtr<Type> m_weak_this;
};

template <typename Type>
SharedPtr<Type>::~SharedPtr() noexcept {
  reset();
}

template <typename Type>
template <typename OtherType, typename Deleter, typename Allocator>
SharedPtr<Type>::SharedPtr(OtherType* ptr, Deleter deleter, Allocator allocator)
  requires std::is_convertible_v<OtherType*, Type*>
{
  reset(ptr, deleter, allocator);
}

template <typename Type>
SharedPtr<Type>::SharedPtr(const SharedPtr& other) : m_ptr(other.m_ptr), m_block(other.m_block) {
  if (m_block != nullptr) {
    ++m_block->shared_cnt;
  }
}

template <typename Type>
template <typename OtherType>
SharedPtr<Type>::SharedPtr(const SharedPtr<OtherType>& other)
  requires std::is_convertible_v<OtherType*, Type*>
    : m_ptr(other.m_ptr), m_block(other.m_block) {
  if (m_block != nullptr) {
    ++m_block->shared_cnt;
  }
}

template <typename Type>
SharedPtr<Type>::SharedPtr(SharedPtr&& other) : m_ptr(other.m_ptr), m_block(other.m_block) {
  other.m_block = nullptr;
  other.m_ptr = nullptr;
}

template <typename Type>
template <typename OtherType>
SharedPtr<Type>::SharedPtr(SharedPtr<OtherType>&& other)
  requires std::is_convertible_v<OtherType*, Type*>
    : m_ptr(other.m_ptr), m_block(other.m_block) {
  other.m_block = nullptr;
  other.m_ptr = nullptr;
}

template <typename Type>
SharedPtr<Type>& SharedPtr<Type>::operator=(const SharedPtr& other) {
  SharedPtr<Type> temp = other;
  swap(temp);
  return *this;
}

template <typename Type>
template <typename OtherType>
SharedPtr<Type>& SharedPtr<Type>::operator=(const SharedPtr<OtherType>& other)
  requires std::is_convertible_v<OtherType*, Type*>
{
  SharedPtr<Type> temp = other;
  swap(temp);
  return *this;
}

template <typename Type>
SharedPtr<Type>& SharedPtr<Type>::operator=(SharedPtr&& other) {
  SharedPtr<Type> temp = std::move(other);
  swap(temp);
  return *this;
}

template <typename Type>
template <typename OtherType>
SharedPtr<Type>& SharedPtr<Type>::operator=(SharedPtr<OtherType>&& other)
  requires std::is_convertible_v<OtherType*, Type*>
{
  SharedPtr<Type> temp = std::move(other);
  swap(temp);
  return *this;
}

template <typename Type>
template <typename OtherType>
SharedPtr<Type>::SharedPtr(const WeakPtr<OtherType>& other)
  requires std::is_convertible_v<OtherType*, Type*>
    : m_ptr(other.m_ptr), m_block(other.m_block) {
  if (!other.expired()) {
    ++m_block->shared_cnt;
  }
}

template <typename Type>
template <typename OtherType, typename Deleter, typename Allocator>
void SharedPtr<Type>::reset(OtherType* ptr, Deleter deleter, Allocator allocator) noexcept
  requires std::is_convertible_v<OtherType*, Type*>
{
  if (m_block != nullptr) {
    --m_block->shared_cnt;
    if (m_block->shared_cnt == 0) {
      m_block->delete_obj();
      if (m_block->weak_cnt == 0) {
        m_block->delete_cls();
      }
    }
    m_ptr = nullptr;
    m_block = nullptr;
  }

  if (ptr != nullptr) {
    using BlockAllocator = typename std::allocator_traits<Allocator>::template rebind_alloc<
        ControlBlockRegular<Deleter, Allocator>>;
    BlockAllocator block_allocator = allocator;
    auto* block = std::allocator_traits<BlockAllocator>::allocate(block_allocator, 1);
    new (block) ControlBlockRegular<Deleter, Allocator>(deleter, allocator);

    block->ptr = ptr;
    ++block->shared_cnt;
    m_ptr = ptr;
    m_block = block;

    if constexpr (std::is_base_of_v<EnableSharedFromThis<OtherType>, OtherType>) {
      ptr->m_weak_this = *this;
    }
  }
}

template <typename Type>
void SharedPtr<Type>::swap(SharedPtr& other) noexcept {
  std::swap(m_ptr, other.m_ptr);
  std::swap(m_block, other.m_block);
}

template <typename Type>
size_t SharedPtr<Type>::use_count() const noexcept {
  return m_block->shared_cnt;
}

template <typename Type>
typename SharedPtr<Type>::element_type* SharedPtr<Type>::get() const noexcept {
  return m_ptr;
}

template <typename Type>
typename SharedPtr<Type>::element_type* SharedPtr<Type>::operator->() const noexcept {
  return m_ptr;
}

template <typename Type>
typename SharedPtr<Type>::element_type& SharedPtr<Type>::operator*() const noexcept {
  return *m_ptr;
}

template <typename Type>
template <typename Deleter, typename Allocator>
SharedPtr<Type>::ControlBlockRegular<Deleter, Allocator>::ControlBlockRegular(
    Deleter n_deleter, Allocator n_allocator) noexcept
    : deleter(n_deleter), allocator(n_allocator) {}

template <typename Type>
template <typename Deleter, typename Allocator>
void SharedPtr<Type>::ControlBlockRegular<Deleter, Allocator>::delete_obj() noexcept {
  deleter(ptr);
  ptr = nullptr;
}

template <typename Type>
template <typename Deleter, typename Allocator>
void SharedPtr<Type>::ControlBlockRegular<Deleter, Allocator>::delete_cls() noexcept {
  using BlockAllocator =
      typename std::allocator_traits<Allocator>::template rebind_alloc<ControlBlockRegular>;
  BlockAllocator block_allocator = allocator;
  std::destroy_at(this);
  std::allocator_traits<BlockAllocator>::deallocate(block_allocator, this, 1);
}

template <typename Type>
template <typename Allocator>
template <typename... Args>
SharedPtr<Type>::ControlBlockShared<Allocator>::ControlBlockShared(Allocator n_allocator,
                                                                   Args&&... args) noexcept
    : obj(std::forward<Args>(args)...), allocator(n_allocator) {}

template <typename Type>
template <typename Allocator>
void SharedPtr<Type>::ControlBlockShared<Allocator>::delete_obj() noexcept {
  std::allocator_traits<Allocator>::destroy(allocator, &obj);
}

template <typename Type>
template <typename Allocator>
void SharedPtr<Type>::ControlBlockShared<Allocator>::delete_cls() noexcept {
  using BlockAllocator =
      typename std::allocator_traits<Allocator>::template rebind_alloc<ControlBlockShared>;
  BlockAllocator block_allocator = allocator;
  std::allocator_traits<BlockAllocator>::deallocate(block_allocator, this, 1);
}

template <typename Type>
WeakPtr<Type>::~WeakPtr() noexcept {
  if (m_block != nullptr) {
    --m_block->weak_cnt;
    if (m_block->shared_cnt == 0 && m_block->weak_cnt == 0) {
      m_block->delete_cls();
    }
  }
}

template <typename Type>
WeakPtr<Type>::WeakPtr(const WeakPtr& other) : m_ptr(other.m_ptr), m_block(other.m_block) {
  if (m_block != nullptr) {
    ++m_block->weak_cnt;
  }
}

template <typename Type>
template <typename OtherType>
WeakPtr<Type>::WeakPtr(const WeakPtr<OtherType>& other)
  requires std::is_convertible_v<OtherType*, Type*>
    : m_ptr(other.m_ptr), m_block(other.m_block) {
  if (m_block != nullptr) {
    ++m_block->weak_cnt;
  }
}

template <typename Type>
WeakPtr<Type>::WeakPtr(WeakPtr&& other) : m_ptr(other.m_ptr), m_block(other.m_block) {
  other.m_ptr = nullptr;
  other.m_block = nullptr;
}

template <typename Type>
template <typename OtherType>
WeakPtr<Type>::WeakPtr(WeakPtr<OtherType>&& other)
  requires std::is_convertible_v<OtherType*, Type*>
    : m_ptr(other.m_ptr), m_block(other.m_block) {
  other.m_ptr = nullptr;
  other.m_block = nullptr;
}

template <typename Type>
template <typename OtherType>
WeakPtr<Type>::WeakPtr(const SharedPtr<OtherType>& other)
  requires std::is_convertible_v<OtherType*, Type*>
    : m_ptr(other.m_ptr), m_block(other.m_block) {
  if (m_block != nullptr) {
    ++m_block->weak_cnt;
  }
}

template <typename Type>
WeakPtr<Type>& WeakPtr<Type>::operator=(const WeakPtr& other) {
  WeakPtr<Type> temp = other;
  swap(temp);
  return *this;
}

template <typename Type>
template <typename OtherType>
WeakPtr<Type>& WeakPtr<Type>::operator=(const WeakPtr<OtherType>& other)
  requires std::is_convertible_v<OtherType*, Type*>
{
  WeakPtr<Type> temp = other;
  swap(temp);
  return *this;
}

template <typename Type>
WeakPtr<Type>& WeakPtr<Type>::operator=(WeakPtr&& other) {
  WeakPtr<Type> temp = std::move(other);
  swap(temp);
  return *this;
}

template <typename Type>
template <typename OtherType>
WeakPtr<Type>& WeakPtr<Type>::operator=(WeakPtr<OtherType>&& other)
  requires std::is_convertible_v<OtherType*, Type*>
{
  WeakPtr<Type> temp = std::move(other);
  swap(temp);
  return *this;
}

template <typename Type>
template <typename OtherType>
WeakPtr<Type>& WeakPtr<Type>::operator=(const SharedPtr<OtherType>& other)
  requires std::is_convertible_v<OtherType*, Type*>
{
  WeakPtr<Type> temp = other;
  swap(temp);
  return *this;
}

template <typename Type>
void WeakPtr<Type>::swap(WeakPtr& other) noexcept {
  std::swap(m_block, other.m_block);
  std::swap(m_ptr, other.m_ptr);
}

template <typename Type>
bool WeakPtr<Type>::expired() const noexcept {
  return use_count() == 0;
}

template <typename Type>
size_t WeakPtr<Type>::use_count() const noexcept {
  return m_block == nullptr ? 0 : m_block->shared_cnt;
}

template <typename Type>
SharedPtr<Type> WeakPtr<Type>::lock() const noexcept {
  return expired() ? SharedPtr<element_type>() : SharedPtr<Type>(*this);
}

template <typename Type>
SharedPtr<Type> EnableSharedFromThis<Type>::shared_from_this() {
  return SharedPtr<Type>(m_weak_this);
}
template <typename Type>
SharedPtr<const Type> EnableSharedFromThis<Type>::shared_from_this() const {
  return SharedPtr<Type>(m_weak_this);
}
template <typename Type>
WeakPtr<Type> EnableSharedFromThis<Type>::weak_from_this() {
  return m_weak_this;
}

template <typename Type>
WeakPtr<const Type> EnableSharedFromThis<Type>::weak_from_this() const {
  return m_weak_this;
}

template <typename Type, typename Allocator, typename... Args>
SharedPtr<Type> allocateShared(const Allocator& allocator, Args&&... args) {
  using BlockAllocator = typename std::allocator_traits<Allocator>::template rebind_alloc<
      typename SharedPtr<Type>::template ControlBlockShared<Allocator>>;

  BlockAllocator block_allocator = allocator;
  auto* block = std::allocator_traits<BlockAllocator>::allocate(block_allocator, 1);
  std::allocator_traits<BlockAllocator>::construct(block_allocator, block, allocator,
                                                   std::forward<Args>(args)...);
  ++block->shared_cnt;

  SharedPtr<Type> shared;
  shared.m_block = block;
  shared.m_ptr = &block->obj;
  return shared;
}

template <typename Type, typename... Args>
SharedPtr<Type> makeShared(Args&&... args) {
  return allocateShared<Type>(std::allocator<Type>(), std::forward<Args>(args)...);
}
