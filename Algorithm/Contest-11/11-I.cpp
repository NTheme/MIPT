/******************************************
 *  Author : NThemeDEV
 *  Created : Mon Oct 16 2023
 *  File : 11-I.cpp
 ******************************************/

/*
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")
*/

#include <algorithm>
#include <iostream>
#include <memory>
#include <set>
#include <vector>

using std::pair;
using std::shared_ptr;
using std::string;
using std::vector;

template <typename TypeFirst, typename TypeSecond>
std::ostream& operator<<(std::ostream& out,
                         const pair<TypeFirst, TypeSecond>& pair) {
  out << pair.first << ' ' << pair.second;
  return out;
}

template <typename Type>
std::istream& operator>>(std::istream& inp, vector<Type>& array) {
  for (auto& elem : array) {
    inp >> elem;
  }
  return inp;
}

template <typename Type>
std::ostream& operator<<(std::ostream& out, const vector<Type>& array) {
  for (const auto& elem : array) {
    out << elem << ' ';
  }
  return out;
}

template <typename Type>
class Stack {
 public:
  Stack() = default;
  ~Stack();

  size_t Size();
  Type Back();
  void Push(const Type& val);
  Type Pop();
  void Clear();

 private:
  struct Element;

  Element* m_root_;
  size_t m_size_;
};

template <typename Type>
struct Stack<Type>::Element {
  Element* m_next;
  Type m_data;

  Element(Element* next, const Type& val);
};

template <typename Type>
Stack<Type>::Element::Element(Element* next, const Type& val)
    : m_next(next), m_data(val) {}

template <typename Type>
Stack<Type>::~Stack() {
  Clear();
}

template <typename Type>
void Stack<Type>::Push(const Type& val) {
  m_root_ = new Element(m_root_, val);
  ++m_size_;
}

template <typename Type>
size_t Stack<Type>::Size() {
  return m_size_;
}

template <typename Type>
Type Stack<Type>::Back() {
  return m_root_->m_data;
}

template <typename Type>
Type Stack<Type>::Pop() {
  Element* del = m_root_;
  m_root_ = m_root_->m_next;
  --m_size_;

  Type removed = del->m_data;
  delete del;

  return removed;
}

template <typename Type>
void Stack<Type>::Clear() {
  while (m_root_ != nullptr) {
    Element* del = m_root_;
    m_root_ = m_root_->m_next;
    --m_size_;
    delete del;
  }
}

template <typename Type>
class Queue {
 public:
  size_t Size();
  void Push(const Type& val);
  Type Pop();
  Type Front();
  void Clear();
  Type Min();

 private:
  Stack<std::pair<Type, Type>> m_left_, m_right_;

  void Balance();
};

template <typename Type>
size_t Queue<Type>::Size() {
  return m_left_.Size() + m_right_.Size();
}

template <typename Type>
void Queue<Type>::Push(const Type& val) {
  Type min_val =
      (m_left_.Size() == 0) ? val : std::min(val, m_left_.Back().second);
  m_left_.Push({val, min_val});
}

template <typename Type>
Type Queue<Type>::Pop() {
  Balance();

  if (m_right_.Size() == 0) {
    return 0;
  }
  return m_right_.Pop().first;
}

template <typename Type>
Type Queue<Type>::Front() {
  Balance();

  if (m_right_.Size() == 0) {
    return 0;
  }
  return m_right_.Back().first;
}

template <typename Type>
void Queue<Type>::Clear() {
  while (m_left_.Size() > 0) {
    m_left_.Pop();
  }
  while (m_right_.Size() > 0) {
    m_right_.Pop();
  }
}

template <typename Type>
Type Queue<Type>::Min() {
  Balance();

  if (Size() == 0) {
    return 0;
  }
  if (m_left_.Size() == 0) {
    return m_right_.Back().second;
  }
  return std::min(m_left_.Back().second, m_right_.Back().second);
}

template <typename Type>
void Queue<Type>::Balance() {
  if (m_right_.Size() == 0) {
    while (m_left_.Size() > 0) {
      Type elem = m_left_.Pop().first;
      int min_val = (m_right_.Size() == 0)
                        ? elem
                        : std::min(elem, m_right_.Back().second);
      m_right_.Push({elem, min_val});
    }
  }
}

class SuffixArray {
 public:
  SuffixArray(const std::string& string, bool cyclic = false);

  bool IsCyclic() const;
  std::vector<size_t> GetArray() const;
  std::vector<size_t> GetClasses() const;
  std::vector<size_t> GetLCP() const;

 private:
  bool m_cyclic_;
  std::vector<size_t> m_array_;
  std::vector<size_t> m_classes_;
  std::vector<size_t> m_lcp_;

  void CountLCP(const string& string);
  void CountArray(const string& string);
};

SuffixArray::SuffixArray(const std::string& string, bool cyclic)
    : m_cyclic_(cyclic),
      m_array_(string.size()),
      m_classes_(string.size()),
      m_lcp_(string.size()) {
  std::string str_copy = string;
  if (!m_cyclic_) {
    str_copy += '\0';
    m_array_.push_back(0);
    m_classes_.push_back(0);
    m_lcp_.push_back(0);
  }

  CountArray(str_copy);
  CountLCP(str_copy);
}

bool SuffixArray::IsCyclic() const { return m_cyclic_; }

std::vector<size_t> SuffixArray::GetArray() const { return m_array_; }

std::vector<size_t> SuffixArray::GetClasses() const { return m_classes_; }

std::vector<size_t> SuffixArray::GetLCP() const { return m_lcp_; }

void SuffixArray::CountArray(const string& string) {
  for (size_t index = 0; index < string.size(); ++index) {
    m_array_[index] = index;
  }
  sort(m_array_.begin(), m_array_.end(),
       [&](size_t left, size_t right) { return string[left] < string[right]; });

  size_t cur = 0;
  vector<size_t> pos(string.size());
  for (size_t index = 0; index < string.size(); index++) {
    if (index == 0 || string[m_array_[index]] != string[m_array_[index - 1]]) {
      pos[cur++] = index;
    }
    m_classes_[m_array_[index]] = cur - 1;
  }

  for (size_t len = 1; len < string.size(); len *= 2) {
    vector<size_t> suf_new(string.size());
    vector<size_t> cls_new(string.size());

    for (size_t index = 0; index < string.size(); index++) {
      size_t next = (m_array_[index] + string.size() - len) % string.size();
      suf_new[pos[m_classes_[next]]++] = next;
    }

    cur = 0;
    for (size_t index = 0; index < string.size(); index++) {
      if (index == 0 ||
          m_classes_[suf_new[index]] != m_classes_[suf_new[index - 1]] ||
          m_classes_[(suf_new[index] + len) % string.size()] !=
              m_classes_[(suf_new[index - 1] + len) % string.size()]) {
        pos[cur++] = index;
      }
      cls_new[suf_new[index]] = cur - 1;
    }
    m_array_ = suf_new;
    m_classes_ = cls_new;
  }
}

void SuffixArray::CountLCP(const string& string) {
  size_t shift = 0;
  for (size_t index = 0; index < string.size(); ++index) {
    if (m_classes_[index] != string.size() - 1) {
      shift -= (shift > 0) ? 1 : 0;
      while (shift < string.size() &&
             string[(m_array_[m_classes_[index]] + shift) % string.size()] ==
                 string[(m_array_[m_classes_[index] + 1] + shift) %
                        string.size()]) {
        ++shift;
      }
      m_lcp_[m_classes_[index]] = shift;
    }
  }
}

int GetID(const vector<int>& id, int xxx) {
  return std::lower_bound(id.begin(), id.end(), xxx) - id.begin() - 1;
}

void Solve() {
  size_t nn;
  std::cin >> nn;

  string str;
  vector<int> id;
  for (size_t i = 0; i < nn; i++) {
    string buf;
    std::cin >> buf;
    id.push_back(str.size() - 1);
    str += buf;
    str.push_back(' ');
  }

  auto arr = SuffixArray(str);
  auto pos = arr.GetArray();
  auto lcp = arr.GetLCP();

  int left = 0;
  int right = -1;
  size_t maximum = 0;
  size_t max_index = 0;
  size_t cnt = 0;
  std::multiset<size_t> lcps;
  vector<size_t> used(nn);
  while (true) {
    if (cnt < nn) {
      ++right;
      if (0 == pos.size() - right) {
        break;
      }
      if (right != 0) {
        lcps.insert(lcp[right - 1]);
      }
      used[GetID(id, pos[right])]++;
      if (used[GetID(id, pos[right])] == 1) {
        cnt++;
      }
    } else {
      lcps.erase(lcps.find(lcp[left]));
      --used[GetID(id, pos[left])];
      if (used[GetID(id, pos[left])] == 0) {
        --cnt;
      }
      ++left;
    }

    if (cnt == nn && *lcps.begin() > maximum && str[pos[left]] != ' ') {
      maximum = *lcps.begin();
      max_index = pos[left];
    }
  }
  std::cout << str.substr(max_index, maximum);
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);

  Solve();

  std::cout.flush();
  return 0;
}
