/******************************************
 *  Author : NThemeDEV
 *  Created : Mon Oct 16 2023
 *  File : 11-D.cpp
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

  if (!m_cyclic_) {
    str_copy.pop_back();
    m_array_.erase(m_array_.begin());
    m_classes_.pop_back();
    m_lcp_.pop_back();
    for (auto& elem : m_classes_) {
      --elem;
    }
  }
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

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);

  string string;
  std::cin >> string;
  auto suff_array = SuffixArray(string, true).GetArray();
  for (size_t index = 0; index < string.size(); ++index) {
    std::cout
        << string[(suff_array[index] + string.size() - 1) % string.size()];
  }
  std::cout.flush();
  return 0;
}
