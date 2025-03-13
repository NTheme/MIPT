#include <algorithm>
#include <iostream>
#include <vector>

class BigInteger {
 public:
  BigInteger();
  BigInteger(long long value);
  BigInteger(const char* string, size_t length);

  BigInteger operator-() const;
  BigInteger& operator+=(const BigInteger& other);
  BigInteger& operator-=(const BigInteger& other);
  BigInteger& operator*=(const BigInteger& other);
  BigInteger& operator/=(const BigInteger& other);
  BigInteger& operator%=(const BigInteger& other);
  BigInteger& operator++();
  BigInteger& operator--();
  BigInteger operator++(int);
  BigInteger operator--(int);

  bool operator==(const BigInteger& other) const;
  bool operator<(const BigInteger& other) const;
  bool operator>(const BigInteger& other) const;
  bool operator<=(const BigInteger& other) const;
  bool operator>=(const BigInteger& other) const;

  explicit operator bool() const;

  void MultiplyToNumber(size_t other);

 private:
  static const size_t kBlockSize = 9;
  static const size_t kBlockValue = 1'000'000'000;

  std::vector<long long> number_;
  bool isPositive_ = false;

  void Increase(const std::vector<long long>& other);
  void Decrease(const std::vector<long long>& other);
  std::pair<BigInteger, BigInteger> Divide(const BigInteger& other) const;
};

BigInteger::BigInteger() : isPositive_(true) {}

BigInteger::BigInteger(long long value) : isPositive_(value >= 0) {
  value = std::abs(value);
  for (; value > 0; value /= kBlockValue) {
    number_.push_back(value % kBlockValue);
  }
}

BigInteger::BigInteger(const char* string, size_t length) : isPositive_(true) {
  if (length == 0) {
    return;
  }
  if (string[0] == '-' && length > 1) {
    isPositive_ = false;
  }

  size_t first = static_cast<size_t>(string[0] == '-' || string[0] == '+');
  for (size_t index = length; index > first;) {
    number_.push_back(0);
    for (size_t power = 1, digits = 0;
         index > first && digits < BigInteger::kBlockSize;
         index--, power *= 10, ++digits) {
      number_.back() += power * (string[index - 1] - '0');
    }
  }

  while (!number_.empty() && number_.back() == 0) {
    number_.pop_back();
  }
}

BigInteger operator""_bi(unsigned long long n) { return BigInteger(n); }

BigInteger operator""_bi(const char* str, size_t length) {
  return BigInteger(str, length);
}

BigInteger BigInteger::operator-() const {
  BigInteger negative(*this);
  if (!negative.number_.empty()) {
    negative.isPositive_ = !negative.isPositive_;
  }
  return negative;
}

BigInteger& BigInteger::operator+=(const BigInteger& other) {
  if (isPositive_ == other.isPositive_) {
    Increase(other.number_);
  } else {
    Decrease(other.number_);
  }
  return *this;
}

BigInteger& BigInteger::operator-=(const BigInteger& other) {
  if (isPositive_ == other.isPositive_) {
    Decrease(other.number_);
  } else {
    Increase(other.number_);
  }
  return *this;
}

BigInteger& BigInteger::operator*=(const BigInteger& other) {
  BigInteger multiply;

  if (!number_.empty() && !other.number_.empty()) {
    isPositive_ = ((isPositive_ ^ other.isPositive_) == 0);

    auto first = number_;
    auto second = other.number_;
    for (size_t index = 1; index < first.size() + second.size(); ++index) {
      size_t index_first = std::min(index, first.size());
      size_t index_second = (index >= first.size()) ? index - first.size() : 0;

      BigInteger add_digits;
      while (index_first > 0 && index_second < second.size()) {
        add_digits += first[--index_first] * second[index_second++];
      }

      BigInteger add;
      add.number_.assign(index - 1, 0);
      for (const auto& block : add_digits.number_) {
        add.number_.push_back(block);
      }
      multiply += add;
    }
  }

  number_ = multiply.number_;
  return *this;
}

BigInteger& BigInteger::operator/=(const BigInteger& other) {
  if (other.number_.empty()) {
    throw std::invalid_argument("Division by zero.");
  }
  auto divided = Divide(other.isPositive_ ? other : -other);
  if (!divided.first.number_.empty()) {
    divided.first.isPositive_ = ((isPositive_ ^ other.isPositive_) == 0);
  }
  *this = divided.first;
  return *this;
}

BigInteger& BigInteger::operator%=(const BigInteger& other) {
  auto divided = Divide(other.isPositive_ ? other : -other);
  if (!divided.second.number_.empty()) {
    divided.second.isPositive_ = this->isPositive_;
  }
  *this = divided.second;
  return *this;
}

BigInteger operator+(const BigInteger& left, const BigInteger& right) {
  BigInteger result = left;
  return result += right;
}

BigInteger operator-(const BigInteger& left, const BigInteger& right) {
  BigInteger result = left;
  return result -= right;
}

BigInteger operator*(const BigInteger& left, const BigInteger& right) {
  BigInteger result = left;
  return result *= right;
}

BigInteger operator/(const BigInteger& left, const BigInteger& right) {
  if (right == 0) {
    throw std::invalid_argument("Division by zero.");
  }
  BigInteger result = left;
  return result /= right;
}

BigInteger operator%(const BigInteger& left, const BigInteger& right) {
  BigInteger result = left;
  return result %= right;
}

BigInteger& BigInteger::operator++() { return *this += 1; }

BigInteger& BigInteger::operator--() { return *this -= 1; }

BigInteger BigInteger::operator++(int) {
  BigInteger old(*this);
  ++*this;
  return old;
}

BigInteger BigInteger::operator--(int) {
  BigInteger old(*this);
  --*this;
  return old;
}
bool BigInteger::operator==(const BigInteger& other) const {
  return isPositive_ == other.isPositive_ && number_ == other.number_;
}

bool BigInteger::operator<(const BigInteger& other) const {
  if (isPositive_ != other.isPositive_) {
    return !isPositive_;
  }

  if (number_.size() != other.number_.size()) {
    return number_.size() * (isPositive_ ? 1 : -1) <
           other.number_.size() * (isPositive_ ? 1 : -1);
  }

  for (size_t index = number_.size(); index > 0; --index) {
    if (number_[index - 1] != other.number_[index - 1]) {
      return number_[index - 1] * (isPositive_ ? 1 : -1) <
             other.number_[index - 1] * (isPositive_ ? 1 : -1);
    }
  }
  return false;
}

bool BigInteger::operator>(const BigInteger& other) const {
  if (isPositive_ != other.isPositive_) {
    return isPositive_;
  }

  if (number_.size() != other.number_.size()) {
    return number_.size() * (isPositive_ ? 1 : -1) >
           other.number_.size() * (isPositive_ ? 1 : -1);
  }

  for (size_t index = number_.size(); index > 0; --index) {
    if (number_[index - 1] != other.number_[index - 1]) {
      return number_[index - 1] * (isPositive_ ? 1 : -1) >
             other.number_[index - 1] * (isPositive_ ? 1 : -1);
    }
  }
  return false;
}
bool BigInteger::operator<=(const BigInteger& other) const {
  return !(other < *this);
}
bool BigInteger::operator>=(const BigInteger& other) const {
  return !(other > *this);
}

BigInteger::operator bool() const { return !number_.empty(); }

void BigInteger::MultiplyToNumber(size_t other) {
  size_t carry = 0;
  for (auto& block : number_) {
    block = block * other + carry;
    carry = block / kBlockValue;
    block %= kBlockValue;
  }

  if (carry > 0) {
    number_.push_back(carry);
  }
}

void BigInteger::Increase(const std::vector<long long>& other) {
  size_t carry = 0;
  for (size_t index = 0; index < other.size(); ++index) {
    if (index == number_.size()) {
      number_.push_back(0);
    }
    number_[index] += carry + other[index];
    carry = number_[index] / kBlockValue;
    number_[index] %= kBlockValue;
  }

  if (carry != 0) {
    if (other.size() == number_.size()) {
      number_.push_back(0);
    }
    number_[other.size()] += carry;
  }
}

void BigInteger::Decrease(const std::vector<long long>& other) {
  for (size_t index = 0; index < other.size(); ++index) {
    if (index == number_.size()) {
      number_.push_back(0);
    }
    number_[index] -= other[index];
  }

  while (!number_.empty() && number_.back() == 0) {
    number_.pop_back();
  }

  if (!number_.empty()) {
    if (number_.back() < 0) {
      isPositive_ = !isPositive_;
      for (auto& block : number_) {
        block = -block;
      }
    }

    for (size_t index = 0; index < number_.size() - 1; ++index) {
      if (number_[index] < 0) {
        number_[index] += kBlockValue;
        --number_[index + 1];
      }
    }

    while (!number_.empty() && number_.back() == 0) {
      number_.pop_back();
    }
  }
}

std::pair<BigInteger, BigInteger> BigInteger::Divide(
    const BigInteger& other) const {
  if (number_.empty() && !other.number_.empty()) {
    return {0, 0};
  }

  BigInteger quotient;
  BigInteger rest;
  BigInteger dividend = isPositive_ ? *this : -*this;
  size_t pow_last = 1;
  size_t div_last = 1;

  for (size_t last_block = dividend.number_.back(); last_block >= 10;
       last_block /= 10) {
    ++pow_last;
    div_last *= 10;
  }

  size_t length = (dividend.number_.size() - 1) * kBlockSize + pow_last;
  for (; length > 0; --length, div_last /= 10) {
    rest.MultiplyToNumber(10);
    if (div_last != 0) {
      rest += dividend.number_.back() / div_last;
      dividend.number_.back() %= div_last;
    }

    if (rest >= other) {
      size_t multiply = 0;
      for (; rest >= other; ++multiply, rest -= other) {
      }

      BigInteger add;
      add.number_.assign(dividend.number_.size() - 1, 0);
      add.number_.push_back(multiply * div_last);
      quotient += add;
    }

    if (length % kBlockSize == 1) {
      div_last = kBlockValue;
      dividend.number_.pop_back();
    }
  }

  return {quotient, rest};
}

std::istream& operator>>(std::istream& inp, BigInteger& number) {
  std::string buffer;
  inp >> buffer;
  number = BigInteger(buffer.c_str(), buffer.size());
  return inp;
}

long long n = 0, k_mod = 1e18;

bool Check(long long k, long long j) {
  long long c = k & j, kn = 0, jn = 0;
  if (c != 0) {
    long long maxn = 0;
    while (c >= (1 << maxn)) {
      maxn++;
    }
    std::vector<long long> mm(maxn, 0);
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
  std::vector<long long> kk(n, 0), jj(n, 0);
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
    std::vector<long long> mm(maxn, 0);
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
                                   BigInteger pw) {
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
  std::ios::sync_with_stdio(false);
  std::cin.tie(0);
  std::cout.tie(0);
  std::cout.precision(20);

  BigInteger m;
  std::cin >> m >> n >> k_mod;

  std::vector<std::vector<long long>> d(1 << n,
                                        std::vector<long long>(1 << n, 0));
  for (long long i = 0; i < 1 << n; i++) {
    for (long long j = 0; j < (1 << n); j++) {
      if (Check(i, j)) {
        d[i][j] = 1;
      } else {
        d[i][j] = 0;
      }
    }
  }

  std::vector<std::vector<long long>> a(1, std::vector<long long>(1 << n, 0));
  for (long long i = 0; i < (1 << n); i++) {
    a[0][i] = 1;
  }

  a = a * Pow(d, m - 1);

  long long ans = 0;
  for (long long i = 0; i < (1 << n); i++) {
    ans = (ans + a[0][i]) % k_mod;
  }

  std::cout << ans;

  return 0;
}