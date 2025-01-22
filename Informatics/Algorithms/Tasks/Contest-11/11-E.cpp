/******************************************
 *  Author : NThemeDEV
 *  Created : Mon Oct 22 2023
 *  File : 11-E.cpp
 ******************************************/

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

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

  void CountLCP(const std::string& string);
  void CountArray(const std::string& string);
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

  if (!m_cyclic_) {
    str_copy.pop_back();
    m_array_.erase(m_array_.begin());
    m_classes_.pop_back();
    for (auto& elem : m_classes_) {
      --elem;
    }
  }
}

bool SuffixArray::IsCyclic() const { return m_cyclic_; }

std::vector<size_t> SuffixArray::GetArray() const { return m_array_; }

std::vector<size_t> SuffixArray::GetClasses() const { return m_classes_; }

std::vector<size_t> SuffixArray::GetLCP() const { return m_lcp_; }

void SuffixArray::CountArray(const std::string& string) {
  for (size_t index = 0; index < string.size(); ++index) {
    m_array_[index] = index;
  }
  sort(m_array_.begin(), m_array_.end(),
       [&](size_t left, size_t right) { return string[left] < string[right]; });

  size_t cur = 0;
  std::vector<size_t> pos(string.size());
  for (size_t index = 0; index < string.size(); index++) {
    if (index == 0 || string[m_array_[index]] != string[m_array_[index - 1]]) {
      pos[cur++] = index;
    }
    m_classes_[m_array_[index]] = cur - 1;
  }

  for (size_t len = 1; len < string.size(); len *= 2) {
    std::vector<size_t> suf_new(string.size());
    std::vector<size_t> cls_new(string.size());

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

void SuffixArray::CountLCP(const std::string& string) {
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

static constexpr size_t kAlphabet = 26;

std::string BWT(const std::string& str) {
  auto suff_array = SuffixArray(str, true).GetArray();
  std::string ret;
  for (size_t index = 0; index < str.size(); ++index) {
    ret += str[(suff_array[index] + str.size() - 1) % str.size()];
  }
  return ret;
}

std::string MTF(const std::string& str) {
  std::vector<uint8_t> dict(kAlphabet);
  for (uint8_t index = 0; index < dict.size(); ++index) {
    dict[index] = index;
  }

  std::string ret(str.size(), '0');
  for (size_t index = 0; index < str.size(); ++index) {
    size_t pos = 0;
    for (; dict[pos] != str[index] - 'a'; ++pos) {
    }
    for (size_t ind = pos; ind > 0; --ind) {
      dict[ind] = dict[ind - 1];
    }
    dict[0] = str[index] - 'a';
    ret[index] = pos + 'a';
  }
  return ret;
}

std::string RLE(const std::string& str) {
  size_t cnt = 1;
  std::string ret(1, str[0]);
  for (size_t index = 1; index < str.size(); ++index, ++cnt) {
    if (str[index] != str[index - 1]) {
      ret += std::to_string(cnt) + str[index];
      cnt = 0;
    }
  }
  ret += std::to_string(cnt);
  return ret;
}

std::string UnBWT(const std::string& str, size_t shift) {
  std::vector<size_t> equal(kAlphabet);
  for (size_t index = 0; index < str.size(); ++index) {
    ++equal[str[index] - 'a'];
  }
  for (uint8_t index = 1; index < equal.size(); ++index) {
    equal[index] += equal[index - 1];
  }

  std::vector<size_t> eq_num(str.size());
  for (size_t index = str.size(); index > 0; --index) {
    eq_num[--equal[str[index - 1] - 'a']] = index - 1;
  }

  std::string ret;
  size_t pos = shift;
  for (size_t index = 0; index < str.size(); ++index) {
    pos = eq_num[pos];
    ret += str[pos];
  }
  return ret;
}

std::string UnMTF(const std::string& str) {
  std::vector<uint8_t> dict(kAlphabet);
  for (uint8_t index = 0; index < dict.size(); ++index) {
    dict[index] = index;
  }

  std::string ret;
  for (size_t index = 0; index < str.size(); ++index) {
    ret += dict[str[index] - 'a'] + 'a';
    uint8_t cur_char = dict[str[index] - 'a'];
    for (uint8_t ind = str[index] - 'a'; ind > 0; --ind) {
      dict[ind] = dict[ind - 1];
    }
    dict[0] = cur_char;
  }
  return ret;
}

std::string UnRLE(const std::string& str) {
  std::string numb;
  std::string ret(1, str[0]);
  for (size_t index = 1; index < str.size(); ++index) {
    if (str[index] <= '9') {
      numb += str[index];
    } else {
      ret += std::string(std::stoul(numb) - 1, ret.back()) + str[index];
      numb.clear();
    }
  }
  ret += std::string(std::stoul(numb) - 1, ret.back());

  return ret;
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);

  uint8_t type;
  std::string str;
  std::cin >> type >> str;

  if (type == '1') {
    std::cout << RLE(MTF(BWT(str)));
  } else if (type == '2') {
    size_t shift;
    std::cin >> shift;
    std::cout << UnBWT(UnMTF(UnRLE(str)), shift);
  }

  std::cout.flush();
  return 0;
}
