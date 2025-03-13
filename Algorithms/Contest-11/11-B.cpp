/******************************************
 *  Author : NThemeDEV
 *  Created : Mon Oct 16 2023
 *  File : 11-B.cpp
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

// static constexpr char kNewLine = '\n';

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

    m_array_ = suf_new;
    cls = cls_new;
  }
}

vector<size_t> SuffixArray::Get() { return m_array_; }

vector<size_t> maximum;
vector<size_t> minimum;
vector<size_t> linkk;
long long len;

size_t Gets(size_t vv) {
  return (vv == linkk[vv] ? vv : linkk[vv] = Gets(linkk[vv]));
}

void Unites(size_t uu, size_t vv) {
  uu = Gets(uu), vv = Gets(vv);
  if (uu == vv) {
    return;
  }
  linkk[uu] = vv;
  minimum[vv] = std::min(minimum[vv], minimum[uu]);
  maximum[vv] = std::max(maximum[vv], maximum[uu]);
  len = std::max(len, (long long)(maximum[vv] - minimum[vv]));
}

void Solve(string& string) {
  SuffixArray arr(string);
  auto suff_mass = arr.Get();
  string.push_back('\0');

  vector<size_t> newnum(string.size());
  for (size_t index = 0; index < string.size(); ++index) {
    newnum[suff_mass[index]] = index;
  }

  long long shift = 0;
  vector<size_t> lcp(string.size());
  for (size_t index = 0; index < string.size() - 1; index++) {
    long long pos = newnum[index];
    while (index + shift < string.size() &&
           suff_mass[pos - 1] + shift < string.size() &&
           string[index + shift] == string[suff_mass[pos - 1] + shift]) {
      shift++;
    }
    lcp[pos] = shift;
    shift = std::max(0LL, shift - 1);
  }

  linkk.resize(string.size() + 1);
  minimum.resize(string.size() + 1);
  maximum.resize(string.size() + 1);

  vector<vector<size_t>> cls(string.size() + 1);
  for (size_t index = 1; index < string.size(); index++) {
    cls[lcp[index]].push_back(index);
  }
  for (size_t index = 0; index < string.size(); index++) {
    linkk[index] = index;
    maximum[index] = minimum[index] = suff_mass[index];
  }

  const long long kFuckCodestyleFirst = -1e18;
  const long long kFuckCodestyleSecond = -1e15;
  len = kFuckCodestyleSecond;

  long long ans = kFuckCodestyleFirst;
  for (long long index = string.size(); index >= 0; index--) {
    for (const auto& uu : cls[index]) {
      Unites(uu, uu - 1);
    }
    ans = std::max(ans, index * index + len + index);
  }

  std::cout << ans;
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);

  string string;
  std::cin >> string;

  Solve(string);

  std::cout.flush();
  return 0;
}
