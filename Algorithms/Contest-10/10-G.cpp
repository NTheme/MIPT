/******************************************
 *  Author : NThemeDEV
 *  Created : Sat Sep 09 2023
 *  File : 10-G.cpp
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
std::istream& operator>>(std::istream& inp, pair<TypeFirst, TypeSecond>& pair) {
  inp >> pair.first >> pair.second;
  return inp;
}
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
    out << elem << '\n';
  }
  return out;
}

template <typename T1, typename T2, typename U1, typename U2>
auto operator+(const pair<T1, T2>& first, const pair<U1, U2>& second) {
  return std::make_pair(first.first + second.first,
                        first.second + second.second);
}

template <typename T1, typename T2, typename U1, typename U2>
auto operator-(const pair<T1, T2>& first, const pair<U1, U2>& second) {
  return std::make_pair(first.first - second.first,
                        first.second - second.second);
}

template <typename T1, typename T2, typename U1, typename U2>
auto operator%(const pair<T1, T2>& first, const pair<U1, U2>& second) {
  return std::make_pair(first.first % second.first,
                        first.second % second.second);
}

template <typename T1, typename T2, typename U1, typename U2>
auto operator*(const pair<T1, T2>& first, const pair<U1, U2>& second) {
  return std::make_pair(first.first * second.first,
                        first.second * second.second);
}

template <typename T1, typename T2>
auto operator*(const pair<T1, T2>& first, long long other) {
  return std::make_pair(first.first * other, first.second * other);
}

bool Comparator(const pair<string, size_t>& first,
                const pair<string, size_t>& second) {
  return first.first.size() > second.first.size();
}

struct Hash {
 public:
  static inline const pair<long long, long long> kMod = {1e9 + 179, 1e9 + 9};
  pair<size_t, size_t> val = {0, 0};

  Hash();
  explicit Hash(const string& str);
  bool operator==(const Hash& other) const;
  std::strong_ordering operator<=>(const Hash& other) const;
  Hash operator+(const Hash& other) const;

 private:
  static const size_t kMax = 1e6;
  static inline vector<pair<size_t, size_t>> power = {};

  static void FillPrime();
};

Hash::Hash() {}

Hash::Hash(const string& str) {
  if (power.empty()) {
    FillPrime();
  }

  for (size_t index = 1; index <= str.size(); ++index) {
    long long next = str[index - 1] - 'a' + 1;
    val = (val + power[index] * next) % kMod;
  }
}

bool Hash::operator==(const Hash& other) const { return val == other.val; }

std::strong_ordering Hash::operator<=>(const Hash& other) const {
  return val <=> other.val;
}

Hash Hash::operator+(const Hash& other) const {
  Hash res;
  res.val = val + other.val;
  return res;
}

void Hash::FillPrime() {
  power.assign(kMax + 1, {1, 1});
  const pair<long long, long long> kPrime = {31, 39};
  for (size_t pow = 1; pow <= kMax; ++pow) {
    power[pow] = (power[pow - 1] * kPrime) % kMod;
  }
}

struct NeededHash {
 public:
  enum SIDE { BEGIN, END };

  Hash m_hash;
  size_t m_index = 0;
  SIDE m_side = BEGIN;

  explicit NeededHash(const string& str, size_t index = 0, SIDE side = BEGIN);

  std::strong_ordering operator<=>(const NeededHash& other) const;
};

NeededHash::NeededHash(const string& str, size_t index, SIDE side)
    : m_hash(str), m_index(index), m_side(side) {}

std::strong_ordering NeededHash::operator<=>(const NeededHash& other) const {
  if (m_hash < other.m_hash) {
    return std::strong_ordering::less;
  }
  if (m_hash > other.m_hash) {
    return std::strong_ordering::greater;
  }
  if (m_index < other.m_index) {
    return std::strong_ordering::less;
  }
  if (m_index > other.m_index) {
    return std::strong_ordering::greater;
  }
  if (m_side < other.m_side) {
    return std::strong_ordering::less;
  }
  if (m_side > other.m_side) {
    return std::strong_ordering::greater;
  }
  return std::strong_ordering::equal;
}

void UpdateNeededHash(const pair<string, size_t>& cur,
                      std::set<NeededHash>& table) {
  for (size_t pos = 0; pos <= 2 * cur.first.size(); ++pos) {
    size_t r_begin = pos / 2 + pos % 2;
    size_t len = std::min(pos / 2, cur.first.size() - r_begin);
    size_t l_begin = pos / 2 - len;

    string l_str = cur.first.substr(l_begin, len);
    string r_str = cur.first.substr(r_begin, len);
    std::reverse(l_str.begin(), l_str.end());

    if (l_str == r_str) {
      if (l_begin > 0) {
        table.emplace(cur.first.substr(0, l_begin), cur.second,
                      NeededHash::BEGIN);
      }
      if (r_begin + len < cur.first.size()) {
        table.emplace(cur.first.substr(r_begin + len, cur.first.size()),
                      cur.second, NeededHash::END);
      }
    }
  }
}

vector<pair<size_t, size_t>> GetPalindromes(
    const vector<pair<string, size_t>>& strings) {
  std::set<NeededHash> table;
  vector<pair<size_t, size_t>> palindromes;

  for (size_t index = 0; index < strings.size(); ++index) {
    NeededHash hash(
        string(strings[index].first.rbegin(), strings[index].first.rend()));

    for (auto it = table.lower_bound(hash);
         it != table.end() && it->m_hash == hash.m_hash; ++it) {
      palindromes.emplace_back(it->m_index + 1, strings[index].second + 1);
      if (it->m_side == NeededHash::END) {
        std::swap(palindromes.back().first, palindromes.back().second);
      }
    }

    UpdateNeededHash(strings[index], table);
  }

  return palindromes;
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);

  size_t num_str;
  std::cin >> num_str;

  vector<pair<string, size_t>> strings(num_str);
  for (size_t index = 0; index < num_str; ++index) {
    std::cin >> strings[index].first;
    strings[index].second = index;
  }
  std::sort(strings.begin(), strings.end(), Comparator);

  auto palindromes = GetPalindromes(strings);
  std::cout << palindromes.size() << '\n' << palindromes;

  std::cout.flush();
  return 0;
}
