/******************************************
 *  Author : NThemeDEV
 *  Created : Thu Oct 05 2023
 *  File : 11-A.cpp
 ******************************************/

/*
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")
*/

#include <algorithm>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

using std::pair;
using std::shared_ptr;
using std::string;
using std::vector;

template <typename TypeFirst, typename TypeSecond>
std::istream& operator>>(std::istream& inp, pair<TypeFirst, TypeSecond>& pair) {
  inp >> pair.first >> pair.second;
  return inp;
}
template <typename TypeFirst, typename TypeSecond>
std::ostream& operator<<(std::ostream& out,
                         const pair<TypeFirst, TypeSecond>& pair) {
  out << pair.first << ' ' << pair.second << '\n';
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
  out << '\n';
  return out;
}

class SuffixArray {
 public:
  SuffixArray(const std::string& string, bool cyclic = false);

  std::vector<size_t> Get();

 private:
  std::vector<size_t> m_array_;
};

SuffixArray::SuffixArray(const std::string& string, bool cyclic)
    : m_array_(string.size()) {
  std::string str_copy = string;
  if (!cyclic) {
    str_copy += '\0';
    m_array_.push_back(0);
  }

  for (size_t index = 0; index < str_copy.size(); ++index) {
    m_array_[index] = index;
  }
  sort(m_array_.begin(), m_array_.end(), [&](size_t left, size_t right) {
    return str_copy[left] < str_copy[right];
  });

  vector<size_t> cls(str_copy.size());
  vector<size_t> pos(str_copy.size());

  size_t cur = 0;
  for (size_t index = 0; index < str_copy.size(); index++) {
    if (index == 0 ||
        str_copy[m_array_[index]] != str_copy[m_array_[index - 1]]) {
      pos[cur++] = index;
    }
    cls[m_array_[index]] = cur - 1;
  }

  for (size_t len = 1; len < str_copy.size(); len *= 2) {
    vector<size_t> suf_new(str_copy.size());
    vector<size_t> cls_new(str_copy.size());

    for (size_t index = 0; index < str_copy.size(); index++) {
      size_t next = (m_array_[index] + str_copy.size() - len) % str_copy.size();
      suf_new[pos[cls[next]]++] = next;
    }

    cur = 0;
    for (size_t index = 0; index < str_copy.size(); index++) {
      if (index == 0 || cls[suf_new[index]] != cls[suf_new[index - 1]] ||
          cls[(suf_new[index] + len) % str_copy.size()] !=
              cls[(suf_new[index - 1] + len) % str_copy.size()]) {
        pos[cur++] = index;
      }
      cls_new[suf_new[index]] = cur - 1;
    }

    swap(m_array_, suf_new);
    swap(cls, cls_new);
  }

  if (!cyclic) {
    m_array_.erase(m_array_.begin());
  }
  for (auto& elem : m_array_) {
    ++elem;
  }
}

vector<size_t> SuffixArray::Get() { return m_array_; }

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);

  string string;
  std::cin >> string;
  SuffixArray arr(string);

  std::cout << arr.Get();

  std::cout.flush();
  return 0;
}
