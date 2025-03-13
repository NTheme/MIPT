/******************************************
 *  Author : NThemeDEV
 *  Created : Thu Sep 21 2023
 *  File : 10-E.cpp
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
using std::unordered_map;
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

template <typename T1, typename T2, typename U1, typename U2>
auto operator+(const pair<T1, T2>& first, const pair<U1, U2>& second) {
  return std::make_pair(first.first + second.first,
                        first.second + second.second);
}

template <typename T1, typename T2, typename U1, typename U2>
auto operator*(const pair<T1, T2>& first, const pair<U1, U2>& second) {
  return std::make_pair(first.first * second.first,
                        first.second * second.second);
}

template <typename T1, typename T2, typename U1, typename U2>
auto operator%(const pair<T1, T2>& first, const pair<U1, U2>& second) {
  return std::make_pair(first.first % second.first,
                        first.second % second.second);
}

static const size_t kMax = 1e6;

struct Hash {
 public:
  static inline const pair<unsigned long long, unsigned long long> kMod = {
      1e9 + 179, 1e9 + 9};
  vector<pair<unsigned long long, unsigned long long>> val;

  explicit Hash(const string& str);
  unsigned long long GetHash(size_t len) const;

 private:
  static inline vector<pair<size_t, size_t>> power;

  static void FillPrime();
};

Hash::Hash(const string& str) : val(str.size() + 1) {
  if (power.empty()) {
    FillPrime();
  }
  for (size_t index = 0; index < str.size(); ++index) {
    pair<unsigned long long, unsigned long long> next = {
        str[index] - '0' + 1, str[str.size() - index - 1] - '0' + 1};
    val[index + 1] = (val[index] + power[index] * next) % kMod;
  }
}

unsigned long long Hash::GetHash(size_t len) const {
  return val[len].first * kMod.second + val[len].second;
}

void Hash::FillPrime() {
  power.assign(kMax + 1, {1, 1});
  const pair<long long, long long> kPrime = {31, 39};
  for (size_t pow = 1; pow < power.size(); ++pow) {
    power[pow] = (power[pow - 1] * kPrime) % kMod;
  }
}

vector<size_t> CalculateClasses(const vector<string>& strings,
                                size_t min_length) {
  vector<unordered_map<unsigned long long, size_t>> hashes(kMax);
  for (const auto& str : strings) {
    auto hash = Hash(str);
    for (size_t ind = 0; ind <= str.size(); ++ind) {
      ++hashes[ind][hash.GetHash(ind)];
    }
  }

  vector<size_t> classes(hashes.size());
  for (size_t index = 0; index < hashes.size(); ++index) {
    for (const auto& cur : hashes[index]) {
      if (cur.second >= min_length) {
        ++classes[index];
      }
    }
  }
  return classes;
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);

  size_t num_abitu;
  size_t min_length;
  size_t num_queries;
  std::cin >> num_abitu >> min_length;

  vector<string> strings(num_abitu);
  std::cin >> strings >> num_queries;

  auto classes = CalculateClasses(strings, min_length);
  while (num_queries-- > 0) {
    size_t len;
    std::cin >> len;
    std::cout << classes[len] << '\n';
  }

  std::cout.flush();
  return 0;
}
