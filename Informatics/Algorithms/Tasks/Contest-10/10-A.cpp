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

const int kMod = 1e9 + 179;
const size_t kMax = 1e6;
int power[kMax + 1] = {1};

void FillPrime() {
  const long long kPrime = 31;

  for (size_t pow = 1; pow <= kMax; ++pow) {
    power[pow] = (power[pow - 1] * kPrime) % kMod;
  }
}

vector<int> GetHash(const string& str) {
  vector<int> hash(str.size() + 1);
  for (size_t let = 1; let <= str.size(); ++let) {
    long long next = str[let - 1] - 'a' + 1;
    hash[let] = (hash[let - 1] + next * power[let]) % kMod;
  }
  return hash;
}

signed main() {
  std::cin.tie(nullptr);
  std::ios_base::sync_with_stdio(false);
  // std::cout.precision(20);

  FillPrime();

  size_t nnn;
  string res;
  std::cin >> nnn >> res;
  vector<int> hash_res = GetHash(res);

  for (size_t ind = 1; ind < nnn; ++ind) {
    string cur;
    std::cin >> cur;
    auto hash_cur = GetHash(cur);

    size_t max_let = 0;
    for (size_t let = 1; let <= std::min(res.size(), cur.size()); ++let) {
      long long hash1 = hash_res[res.size()] - hash_res[res.size() - let];
      long long hash2 = hash_cur[let] * (long long)power[res.size() - let];
      if ((hash1 + kMod) % kMod == (hash2 + kMod) % kMod) {
        max_let = let;
      }
    }

    res += cur.substr(max_let, cur.size() - max_let);
    for (size_t let = max_let + 1; let < hash_cur.size(); ++let) {
      long long hash = hash_cur[let] - hash_cur[let - 1] + kMod;
      hash_res.push_back(
          (hash * power[hash_res.size() - let] + hash_res.back()) % kMod);
    }
  }

  std::cout << res << '\n';

  std::cout.flush();
  return 0;
}
