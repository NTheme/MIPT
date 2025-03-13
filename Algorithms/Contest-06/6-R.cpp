#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

long long n = 0, k_mod = 1e18;

bool Check(long long k, long long j) {
  long long c = k & j, kn = 0, jn = 0;
  if (c != 0) {
    long long maxn = 0;
    while (c >= (1 << maxn)) {
      maxn++;
    }
    vector<long long> mm(maxn, 0);
    for (long long i = maxn - 1; i >= 0; i--) {
      if (c >= (1 << i)) {
        mm[i] = 1, c -= (1 << i);
      }
    }
    for (long long i = 1; i < maxn; i++) {
      if (mm[i] == 1 && mm[i - 1] == 1) {
        return false;
      }
    }
  }
  vector<long long> kk(n, 0), jj(n, 0);
  for (long long i = n; i >= 0; i--) {
    if (k >= (1 << i)) {
      kk[i] = 1, k -= (1 << i);
    }
  }
  for (long long i = 0; i < n; i++) {
    if (kk[i] == 0) {
      kn += (1 << i);
    }
  }
  for (long long i = n; i >= 0; i--) {
    if (j >= (1 << i)) {
      jj[i] = 1, j -= (1 << i);
    }
  }

  for (long long i = 0; i < n; i++) {
    if (jj[i] == 0) {
      jn += (1 << i);
    }
  }
  long long q = kn & jn, maxn = 0;
  if (q != 0) {
    while (q >= (1 << maxn)) {
      maxn++;
    }
    vector<long long> mm(maxn, 0);
    for (long long i = maxn - 1; i >= 0; i--) {
      if (q >= (1 << i)) {
        mm[i] = 1, q -= (1 << i);
      }
    }
    for (long long i = 1; i < maxn; i++) {
      if (mm[i] == 1 && mm[i - 1] == 1) {
        return false;
      }
    }
  }
  return true;
}

template <typename Type>
std::vector<std::vector<Type>> operator*(
    const std::vector<std::vector<Type>>& left,
    const std::vector<std::vector<Type>>& right) {
  std::vector<std::vector<Type>> result(left.size(),
                                        std::vector<Type>(right[0].size()));
  for (size_t line = 0; line < left.size(); ++line) {
    for (size_t column = 0; column < right[0].size(); ++column) {
      for (size_t index = 0; index < right.size(); ++index) {
        result[line][column] =
            (result[line][column] + left[line][index] * right[index][column]) %
            k_mod;
      }
    }
  }
  return result;
}

template <typename Type>
std::vector<std::vector<Type>> Pow(const std::vector<std::vector<Type>>& matrix,
                                   unsigned long long pw) {
  std::vector<std::vector<Type>> m_pw = matrix;
  std::vector<std::vector<Type>> res(matrix.size(),
                                     std::vector<Type>(matrix.size()));
  for (size_t i = 0; i < res.size(); ++i) {
    res[i][i] = 1;
  }

  while (pw > 0) {
    if (pw % 2 == 1) {
      res = res * m_pw;
    }
    m_pw = m_pw * m_pw;
    pw /= 2;
  }

  return res;
}

signed main() {
  ios::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
  cout.precision(20);

  long long m = 0;
  cin >> m >> n;
  if (n > m) {
    std::swap(m, n);
  }

  vector<vector<long long>> d(1 << n, vector<long long>(1 << n, 0));
  for (long long i = 0; i < 1 << n; i++) {
    for (long long j = 0; j < (1 << n); j++) {
      if (Check(i, j)) {
        d[i][j] = 1;
      } else {
        d[i][j] = 0;
      }
    }
  }

  vector<vector<long long>> a(1, vector<long long>(1 << n, 0));
  for (long long i = 0; i < (1 << n); i++) {
    a[0][i] = 1;
  }

  a = a * Pow(d, m - 1);

  long long ans = 0;
  for (long long i = 0; i < (1 << n); i++) {
    ans = (ans + a[0][i]) % k_mod;
  }

  cout << ans;

  return 0;
}
