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
    out << elem << ' ';
  }
  return out;
}

template <typename T1, typename T2, typename U1, typename U2>
auto operator+(const pair<T1, U1>& first, const pair<T2, U2>& second) {
  return std::make_pair(first.first + second.first,
                        first.second + second.second);
}

template <typename T1, typename T2, typename U1, typename U2>
auto operator-(const pair<T1, U1>& first, const pair<T2, U2>& second) {
  return std::make_pair(first.first - second.first,
                        first.second - second.second);
}

template <typename T1, typename T2, typename U1, typename U2>
auto operator%(const pair<T1, U1>& first, const pair<T2, U2>& second) {
  return std::make_pair(first.first % second.first,
                        first.second % second.second);
}

template <typename T1, typename T2, typename U1, typename U2>
auto operator*(const pair<T1, U1>& first, const pair<T2, U2>& second) {
  return std::make_pair(first.first * second.first,
                        first.second * second.second);
}

template <typename T1, typename T2, typename U1, typename U2>
bool operator==(const pair<T1, U1>& first, const pair<T2, U2>& second) {
  return first.first == second.first && first.second == second.second;
}

const pair<long long, long long> kMod = {1e9 + 179, 1e9 + 9};
vector<pair<int, int>> power;

void FillPrime() {
  const pair<long long, long long> kPrime = {31, 39};
  const size_t kMax = 2e6 + 1;
  power.assign(kMax, {1, 1});

  for (size_t pow = 1; pow < kMax; ++pow) {
    power[pow] = {(power[pow - 1].first * kPrime.first) % kMod.first,
                  (power[pow - 1].second * kPrime.second) % kMod.second};
  }
}

vector<pair<int, int>> GetHash(const string& str) {
  vector<pair<int, int>> hash(str.size() + 1);
  for (size_t let = 1; let <= str.size(); ++let) {
    long long next = str[let - 1] - 'a' + 1;
    hash[let] = {
        (hash[let - 1].first + next * power[let].first) % kMod.first,
        (hash[let - 1].second + next * power[let].second) % kMod.second};
  }
  return hash;
}

bool Compare(const pair<int, int>& first, const pair<int, int>& second) {
  return first.first < second.first;
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);
  // std::cout.precision(20);

  FillPrime();

  string ss;
  string tt;
  std::cin >> ss >> tt;
  ss += ss;

  auto min_p = std::max(ss.size() / 2, tt.size());
  auto hash_s = GetHash(ss);
  auto hash_t = GetHash(tt);

  vector<pair<int, int>> list;
  for (size_t ind = 0; ind < ss.size() / 2; ++ind) {
    auto hash = hash_s[ind + ss.size() / 2] - hash_s[ind] + kMod;
    list.push_back((hash * power[min_p - ind]) % kMod);
  }
  std::sort(list.begin(), list.end());

  size_t ans = 0;
  for (size_t ind = 0; ind + ss.size() / 2 <= tt.size(); ++ind) {
    auto hash =
        (hash_t[ind + ss.size() / 2] - hash_t[ind] + kMod) * power[min_p - ind];
    auto iter =
        std::lower_bound(list.begin(), list.end(), hash % kMod, Compare);
    ans += (iter != list.end() && *iter == hash % kMod) ? 1 : 0;
  }
  std::cout << ans << '\n';

  std::cout.flush();
  return 0;
}
