#pragma once

#pragma once

#include <algorithm>
#include <array>
#include <compare>
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
  std::strong_ordering operator<=>(const BigInteger& other) const;

  std::string toString() const;
  explicit operator bool() const;

  void swap(BigInteger& other);
  bool positive() const;
  BigInteger absolute() const;
  BigInteger gcd(BigInteger other) const;
  BigInteger lcm(const BigInteger& other) const;
  void multiplyToNumber(size_t other);

 private:
  static const size_t kBlockSize = 9;
  static const size_t kBlockValue = 1'000'000'000;

  std::vector<long long> number;
  bool isPositive = false;

  void increase(const std::vector<long long>& other);
  void decrease(const std::vector<long long>& other);
  std::pair<BigInteger, BigInteger> divide(const BigInteger& other) const;
};

class Rational {
 public:
  Rational();
  Rational(long long value);
  Rational(const BigInteger& value);

  std::string toString() const;
  std::string asDecimal(size_t precision = 0) const;
  explicit operator double() const;

  Rational operator-() const;
  Rational& operator+=(const Rational& other);
  Rational& operator-=(const Rational& other);
  Rational& operator*=(const Rational& other);
  Rational& operator/=(const Rational& other);

  bool operator==(const Rational& other) const;
  std::strong_ordering operator<=>(const BigInteger& other) const;

 private:
  static const size_t kDoublePrecision = 20;

  BigInteger numer;
  BigInteger denom;

  void toFraction();
};

BigInteger::BigInteger() : isPositive(true) {}

BigInteger::BigInteger(long long value) : isPositive(value >= 0) {
  value = std::abs(value);
  for (; value > 0; value /= kBlockValue) {
    number.push_back(value % kBlockValue);
  }
}

BigInteger::BigInteger(const char* string, size_t length) : isPositive(true) {
  if (length == 0) {
    return;
  }
  if (string[0] == '-' && length > 1) {
    isPositive = false;
  }

  size_t first = static_cast<size_t>(string[0] == '-' || string[0] == '+');
  for (size_t index = length; index > first;) {
    number.push_back(0);
    for (size_t power = 1, digits = 0;
         index > first && digits < BigInteger::kBlockSize;
         index--, power *= 10, ++digits) {
      number.back() += power * (string[index - 1] - '0');
    }
  }

  while (!number.empty() && number.back() == 0) {
    number.pop_back();
  }
}

BigInteger operator""_bi(unsigned long long n) { return BigInteger(n); }

BigInteger operator""_bi(const char* str, size_t length) {
  return BigInteger(str, length);
}

BigInteger BigInteger::operator-() const {
  BigInteger negative(*this);
  if (!negative.number.empty()) {
    negative.isPositive = !negative.isPositive;
  }
  return negative;
}

BigInteger& BigInteger::operator+=(const BigInteger& other) {
  if (isPositive == other.isPositive) {
    increase(other.number);
  } else {
    decrease(other.number);
  }
  return *this;
}

BigInteger& BigInteger::operator-=(const BigInteger& other) {
  if (isPositive == other.isPositive) {
    decrease(other.number);
  } else {
    increase(other.number);
  }
  return *this;
}

BigInteger& BigInteger::operator*=(const BigInteger& other) {
  BigInteger multiply;

  if (!number.empty() && !other.number.empty()) {
    isPositive = ((isPositive ^ other.isPositive) == 0);

    auto first = number;
    auto second = other.number;
    for (size_t index = 1; index < first.size() + second.size(); ++index) {
      size_t indexFirst = std::min(index, first.size());
      size_t indexSecond = (index >= first.size()) ? index - first.size() : 0;

      BigInteger addDigits;
      while (indexFirst > 0 && indexSecond < second.size()) {
        addDigits += first[--indexFirst] * second[indexSecond++];
      }

      BigInteger add;
      add.number.assign(index - 1, 0);
      for (const auto& block : addDigits.number) {
        add.number.push_back(block);
      }
      multiply += add;
    }
  }

  number = multiply.number;
  return *this;
}

BigInteger& BigInteger::operator/=(const BigInteger& other) {
  if (other.number.empty()) {
    throw std::invalid_argument("Division by zero.");
  }
  auto divided = divide(other.absolute());
  if (!divided.first.number.empty()) {
    divided.first.isPositive = ((isPositive ^ other.isPositive) == 0);
  }
  *this = divided.first;
  return *this;
}

BigInteger& BigInteger::operator%=(const BigInteger& other) {
  auto divided = divide(other.absolute());
  if (!divided.second.number.empty()) {
    divided.second.isPositive = this->isPositive;
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
  return isPositive == other.isPositive && number == other.number;
}

std::strong_ordering BigInteger::operator<=>(const BigInteger& other) const {
  if (isPositive != other.isPositive) {
    return isPositive <=> other.isPositive;
  }
  if (number.size() != other.number.size()) {
    return number.size() * (isPositive ? 1 : -1) <=>
           other.number.size() * (isPositive ? 1 : -1);
  }

  for (size_t index = number.size(); index > 0; --index) {
    if (number[index - 1] != other.number[index - 1]) {
      return number[index - 1] * (isPositive ? 1 : -1) <=>
             other.number[index - 1] * (isPositive ? 1 : -1);
    }
  }
  return std::strong_ordering::equal;
}

std::string BigInteger::toString() const {
  if (number.empty()) {
    return "0";
  }

  std::string str_number;
  if (!isPositive) {
    str_number.push_back('-');
  }
  str_number += std::to_string(*number.rbegin());

  for (auto it = number.rbegin() + 1; it != number.rend(); ++it) {
    std::string part = std::to_string(*it);
    str_number += std::string(kBlockSize - part.size(), '0') + part;
  }

  return str_number;
}

BigInteger::operator bool() const { return !number.empty(); }

void BigInteger::swap(BigInteger& other) {
  std::swap(isPositive, other.isPositive);
  std::swap(number, other.number);
}

bool BigInteger::positive() const { return isPositive; }

BigInteger BigInteger::absolute() const { return isPositive ? *this : -*this; }

BigInteger BigInteger::gcd(BigInteger other) const {
  BigInteger left = absolute();
  while (other > 0) {
    left %= other;
    left.swap(other);
  }
  return left;
}

BigInteger BigInteger::lcm(const BigInteger& other) const {
  return *this * other / gcd(other);
}

void BigInteger::multiplyToNumber(size_t other) {
  size_t carry = 0;
  for (auto& block : number) {
    block = block * other + carry;
    carry = block / kBlockValue;
    block %= kBlockValue;
  }

  if (carry > 0) {
    number.push_back(carry);
  }
}

void BigInteger::increase(const std::vector<long long>& other) {
  size_t carry = 0;
  for (size_t index = 0; index < other.size(); ++index) {
    if (index == number.size()) {
      number.push_back(0);
    }
    number[index] += carry + other[index];
    carry = number[index] / kBlockValue;
    number[index] %= kBlockValue;
  }

  if (carry != 0) {
    if (other.size() == number.size()) {
      number.push_back(0);
    }
    number[other.size()] += carry;
  }
}

void BigInteger::decrease(const std::vector<long long>& other) {
  for (size_t index = 0; index < other.size(); ++index) {
    if (index == number.size()) {
      number.push_back(0);
    }
    number[index] -= other[index];
  }

  while (!number.empty() && number.back() == 0) {
    number.pop_back();
  }

  if (!number.empty()) {
    if (number.back() < 0) {
      isPositive = !isPositive;
      for (auto& block : number) {
        block = -block;
      }
    }

    for (size_t index = 0; index < number.size() - 1; ++index) {
      if (number[index] < 0) {
        number[index] += kBlockValue;
        --number[index + 1];
      }
    }

    while (!number.empty() && number.back() == 0) {
      number.pop_back();
    }
  }
}

std::pair<BigInteger, BigInteger> BigInteger::divide(
    const BigInteger& other) const {
  if (number.empty() && !other.number.empty()) {
    return {0, 0};
  }

  BigInteger quotient;
  BigInteger rest;
  BigInteger dividend = absolute();
  size_t powLast = 1;
  size_t divLast = 1;

  for (size_t lastBlock = dividend.number.back(); lastBlock >= 10;
       lastBlock /= 10) {
    ++powLast;
    divLast *= 10;
  }

  size_t length = (dividend.number.size() - 1) * kBlockSize + powLast;
  for (; length > 0; --length, divLast /= 10) {
    rest.multiplyToNumber(10);
    rest += dividend.number.back() / divLast;
    dividend.number.back() %= divLast;

    if (rest >= other) {
      size_t multiply = 0;
      for (; rest >= other; ++multiply, rest -= other) {
      }

      BigInteger add;
      add.number.assign(dividend.number.size() - 1, 0);
      add.number.push_back(multiply * divLast);
      quotient += add;
    }

    if (length % kBlockSize == 1) {
      divLast = kBlockValue;
      dividend.number.pop_back();
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

std::ostream& operator<<(std::ostream& out, const BigInteger& number) {
  return out << number.toString();
}

Rational::Rational() : numer(0), denom(1) {}

Rational::Rational(long long value) : numer(value), denom(1) {}

Rational::Rational(const BigInteger& value) : numer(value), denom(1) {}

std::string Rational::toString() const {
  std::string number = numer.toString();
  if (denom != 1) {
    number += '/' + denom.toString();
  }
  return number;
}

std::string Rational::asDecimal(size_t precision) const {
  std::string number;
  number += (numer / denom).toString();
  if (!numer.positive() && number[0] != '-') {
    number.insert(number.begin(), '-');
  }

  if (precision > 0) {
    number.push_back('.');

    BigInteger numerNew = numer.absolute();
    while (precision-- > 0) {
      numerNew %= denom;
      numerNew.multiplyToNumber(10);
      number += (numerNew / denom).toString();
    }
  }
  return number;
}

Rational::operator double() const {
  return std::stod(asDecimal(kDoublePrecision));
}

Rational Rational::operator-() const {
  Rational negative = *this;
  negative.numer = -numer;
  return negative;
}

Rational& Rational::operator+=(const Rational& other) {
  numer = numer * other.denom + denom * other.numer;
  denom *= other.denom;
  toFraction();
  return *this;
}

Rational& Rational::operator-=(const Rational& other) {
  numer = numer * other.denom - denom * other.numer;
  denom *= other.denom;
  toFraction();
  return *this;
}

Rational& Rational::operator*=(const Rational& other) {
  numer *= other.numer;
  denom *= other.denom;
  toFraction();
  return *this;
}

Rational& Rational::operator/=(const Rational& other) {
  if (other.numer == 0) {
    throw std::invalid_argument("Division by zero.");
  }
  numer *= other.denom;
  denom *= other.numer;
  toFraction();
  return *this;
}

Rational operator+(const Rational& left, const Rational& right) {
  Rational result = left;
  return result += right;
}

Rational operator-(const Rational& left, const Rational& right) {
  Rational result = left;
  return result -= right;
}

Rational operator*(const Rational& left, const Rational& right) {
  Rational result = left;
  return result *= right;
}

Rational operator/(const Rational& left, const Rational& right) {
  if (right == 0) {
    throw std::invalid_argument("Division by zero.");
  }
  Rational result = left;
  return result /= right;
}

bool Rational::operator==(const Rational& other) const {
  return numer == other.numer && denom == other.denom;
}

bool operator!=(const Rational& left, const Rational& right) {
  return !(left == right);
}

std::strong_ordering Rational::operator<=>(const BigInteger& other) const {
  return (*this - other).numer <=> 0;
}

void Rational::toFraction() {
  if (!denom.positive()) {
    numer = -numer;
    denom = -denom;
  }

  BigInteger commonDivisor = numer.gcd(denom);
  numer /= commonDivisor;
  denom /= commonDivisor;
}

std::istream& operator>>(std::istream& inp, Rational& number) {
  BigInteger buffer;
  inp >> buffer;
  number = buffer;
  return inp;
}

std::ostream& operator<<(std::ostream& out, const Rational& number) {
  return out << number.toString();
}

template <size_t N, size_t L, size_t R>
struct FindSqrt {
  static const size_t value = ((L + R + 1) / 2) * ((L + R + 1) / 2) <= N
                                  ? FindSqrt<N, (L + R + 1) / 2, R>::value
                                  : FindSqrt<N, L, (L + R - 1) / 2>::value;
};

template <size_t N, size_t V>
struct FindSqrt<N, V, V> {
  static const size_t value = V;
};

template <size_t N>
struct Sqrt {
  static const size_t value = FindSqrt<N, 0, N>::value;
};

template <size_t N, size_t D>
struct CheckPrime {
  static const bool value = N % D != 0 || CheckPrime<N, D - 1>::value;
};

template <size_t N>
struct CheckPrime<N, 1> {
  static const bool value = (N == 2 || N == 3);
};

template <size_t N>
struct IsPrime {
  static const bool value = CheckPrime<N, Sqrt<N>::value>::value;
};

template <>
struct IsPrime<0> {
  static const bool value = false;
};

template <>
struct IsPrime<1> {
  static const bool value = false;
};

template <size_t N>
class Residue {
 public:
  Residue();
  explicit Residue(int valNew);
  explicit operator int() const;

  bool operator==(const Residue<N>& other) const = default;

  Residue operator-() const;
  Residue& operator+=(const Residue<N>& other);
  Residue& operator-=(const Residue<N>& other);
  Residue& operator*=(const Residue<N>& other);
  Residue& operator/=(const Residue<N>& other);
  Residue& operator++();
  Residue& operator--();
  Residue operator++(int);
  Residue operator--(int);

 private:
  size_t value;

  size_t modPow(size_t value);
};

template <size_t N, size_t M, typename Field = Rational>
class Matrix {
 public:
  Matrix();
  template <typename T>
  Matrix(std::initializer_list<std::initializer_list<T>> data);

  bool operator==(const Matrix<N, M, Field>& other) const;

  std::array<Field, M>& operator[](size_t row);
  const std::array<Field, M>& operator[](size_t row) const;

  Matrix& operator+=(const Matrix<N, M, Field>& other);
  Matrix& operator-=(const Matrix<N, M, Field>& other);
  Matrix& operator*=(const Field& coeff);

  void invert();
  Matrix<N, M, Field> inverted() const;
  Matrix<M, N, Field> transposed() const;
  Field det() const;
  size_t rank() const;
  Field trace() const;

  std::array<Field, M> getRow(size_t row) const;
  std::array<Field, N> getColumn(size_t row) const;

 private:
  enum class UpdateCond;

  std::array<std::array<Field, M>, N> data;

  UpdateCond isInitialized;
  mutable size_t rankVal;
  mutable Field detVal;

  void update(UpdateCond condition) const;
};

template <size_t N, typename Field = Rational>
using SquareMatrix = Matrix<N, N, Field>;

namespace Gaussian {
template <size_t N, size_t M, typename Field = Rational>
static void straight(Matrix<N, M, Field>& copy) {
  for (size_t line = 0, column = 0; line < N && column < M; ++line, ++column) {
    size_t firstNonZero = line;
    while (firstNonZero < N && copy[firstNonZero][column] == Field(0)) {
      ++firstNonZero;
    }

    if (firstNonZero == N) {
      --line;
      continue;
    }

    if (firstNonZero != line) {
      for (size_t index = column; index < M; ++index) {
        copy[line][index] *= Field(-1);
      }
      std::swap(copy[line], copy[firstNonZero]);
    }

    for (size_t indexLine = line + 1; indexLine < N; ++indexLine) {
      Field headMultiply = copy[indexLine][column] / copy[line][column];
      for (size_t indexColumn = column; indexColumn < M; ++indexColumn) {
        copy[indexLine][indexColumn] -= headMultiply * copy[line][indexColumn];
      }
    }
  }
}

template <size_t N, size_t M, typename Field = Rational>
static void revert(Matrix<N, M, Field>& copy) {
  for (size_t column = 0; column < N; ++column) {
    Field headDivide = copy[column][column];
    for (size_t index = column; index < M; ++index) {
      copy[column][index] /= headDivide;
    }

    for (size_t line = column; line > 0; --line) {
      Field multiply = copy[line - 1][column];
      for (size_t index = column; index < M; ++index) {
        copy[line - 1][index] -= multiply * copy[column][index];
      }
    }
  }
}
};  // namespace Gaussian

template <size_t N>
Residue<N>::Residue() : value(0) {}

template <size_t N>
Residue<N>::Residue(int valNew)
    : value(((valNew < 0) ? (N - (-valNew % N)) : valNew) % N) {}

template <size_t N>
Residue<N> Residue<N>::operator-() const {
  return Residue<N>((N - value) % N);
}

template <size_t N>
Residue<N>& Residue<N>::operator+=(const Residue<N>& other) {
  value = (value + other.value) % N;
  return *this;
}

template <size_t N>
Residue<N>& Residue<N>::operator-=(const Residue<N>& other) {
  value = (value + N - other.value) % N;
  return *this;
}

template <size_t N>
Residue<N>& Residue<N>::operator*=(const Residue<N>& other) {
  value = (static_cast<long long>(value) * other.value) % N;
  return *this;
}

template <size_t N>
Residue<N>& Residue<N>::operator/=(const Residue<N>& other) {
  static_assert(IsPrime<N>::value);
  size_t inversed = modPow(other.value);
  value = (value * inversed) % N;
  return *this;
}

template <size_t N>
Residue<N> operator+(const Residue<N>& left, const Residue<N>& right) {
  Residue<N> result = left;
  return result += right;
}

template <size_t N>
Residue<N> operator-(const Residue<N>& left, const Residue<N>& right) {
  Residue<N> result = left;
  return result -= right;
}

template <size_t N>
Residue<N> operator*(const Residue<N>& left, const Residue<N>& right) {
  Residue<N> result = left;
  return result *= right;
}

template <size_t N>
Residue<N> operator/(const Residue<N>& left, const Residue<N>& right) {
  Residue<N> result = left;
  return result /= right;
}

template <size_t N>
size_t Residue<N>::modPow(size_t value) {
  size_t inversed = 1;
  for (size_t power = N - 2; power > 0;
       value = (static_cast<long long>(value) * value) % N, power /= 2) {
    if (power % 2 == 1) {
      inversed = (inversed * value) % N;
    }
  }
  return inversed;
}

template <size_t N>
Residue<N>& Residue<N>::operator++() {
  value = ++value % N;
  return *this;
}

template <size_t N>
Residue<N>& Residue<N>::operator--() {
  value = (--value + N) % N;
  return *this;
}

template <size_t N>
Residue<N> Residue<N>::operator++(int) {
  Residue<N> old(value);
  ++*this;
  return old;
}

template <size_t N>
Residue<N> Residue<N>::operator--(int) {
  Residue<N> old(value);
  --*this;
  return old;
}

template <size_t N>
Residue<N>::operator int() const {
  return static_cast<int>(value);
}

template <size_t N>
std::ostream& operator<<(std::ostream& out, const Residue<N>& value) {
  return out << static_cast<int>(value);
}

template <size_t N, size_t M, typename Field>
enum class Matrix<N, M, Field>::UpdateCond {
  NotInitialized,
  Initialized
};

template <size_t N, size_t M, typename Field>
Matrix<N, M, Field>::Matrix()
    : data({}),
      isInitialized(UpdateCond::NotInitialized),
      rankVal(0),
      detVal(0) {}

template <size_t N, size_t M, typename Field>
template <typename DataType>
Matrix<N, M, Field>::Matrix(
    std::initializer_list<std::initializer_list<DataType>> dataNew)
    : data({}), isInitialized(UpdateCond::NotInitialized) {
  size_t line = 0;
  for (const auto& lineData : dataNew) {
    size_t column = 0;
    for (const auto& value : lineData) {
      data[line][column++] = Field(value);
    }
    ++line;
  }
}

template <size_t N, size_t M, typename Field>
bool Matrix<N, M, Field>::operator==(const Matrix<N, M, Field>& other) const {
  return data == other.data;
}

template <size_t N, size_t M, typename Field>
std::array<Field, M>& Matrix<N, M, Field>::operator[](size_t line) {
  return data[line];
}

template <size_t N, size_t M, typename Field>
const std::array<Field, M>& Matrix<N, M, Field>::operator[](size_t line) const {
  return data[line];
}

template <size_t N, size_t M, typename Field>
Matrix<N, M, Field>& Matrix<N, M, Field>::operator+=(
    const Matrix<N, M, Field>& other) {
  for (size_t line = 0; line < N; ++line) {
    for (size_t column = 0; column < M; ++column) {
      data[line][column] += other.data[line][column];
    }
  }
  update(UpdateCond::Initialized);
  return *this;
}

template <size_t N, size_t M, typename Field>
Matrix<N, M, Field>& Matrix<N, M, Field>::operator-=(
    const Matrix<N, M, Field>& other) {
  for (size_t line = 0; line < N; ++line) {
    for (size_t column = 0; column < M; ++column) {
      data[line][column] -= other.data[line][column];
    }
  }
  update(UpdateCond::Initialized);
  return *this;
}

template <size_t N, size_t M, typename Field>
Matrix<N, M, Field>& Matrix<N, M, Field>::operator*=(const Field& coeff) {
  for (size_t line = 0; line < N; ++line) {
    for (size_t column = 0; column < M; ++column) {
      data[line][column] *= coeff;
    }
  }
  update(UpdateCond::Initialized);
  return *this;
}

template <size_t N, size_t M, typename Field>
Matrix<N, M, Field> operator+(const Matrix<N, M, Field>& left,
                              const Matrix<N, M, Field>& right) {
  Matrix<N, M, Field> result = left;
  return left += right;
}

template <size_t N, size_t M, typename Field>
Matrix<N, M, Field> operator-(const Matrix<N, M, Field>& left,
                              const Matrix<N, M, Field>& right) {
  Matrix<N, M, Field> result = left;
  return result -= right;
}

template <size_t N, size_t M, typename Field>
Matrix<N, M, Field> operator*(const Matrix<N, M, Field>& matrix,
                              const Field& coeff) {
  Matrix<N, M, Field> result = matrix;
  return result *= coeff;
}

template <size_t N, size_t M, typename Field>
Matrix<N, M, Field> operator*(const Field& coeff,
                              const Matrix<N, M, Field>& matrix) {
  return matrix * coeff;
}

template <size_t N, size_t M, size_t K, typename Field>
Matrix<N, K, Field> operator*(const Matrix<N, M, Field>& left,
                              const Matrix<M, K, Field>& right) {
  Matrix<N, K, Field> result;
  for (size_t line = 0; line < N; ++line) {
    for (size_t column = 0; column < K; ++column) {
      for (size_t index = 0; index < M; ++index) {
        result[line][column] += left[line][index] * right[index][column];
      }
    }
  }

  return result;
}

template <size_t N, typename Field>
Matrix<N, N, Field>& operator*=(Matrix<N, N, Field>& left,
                                const Matrix<N, N, Field>& right) {
  return left = left * right;
}

template <size_t N, size_t M, typename Field>
void Matrix<N, M, Field>::invert() {
  static_assert(N == M);

  Matrix<N, 2 * N, Field> copy;
  for (size_t line = 0; line < N; ++line) {
    for (size_t column = 0; column < N; ++column) {
      copy[line][column] = data[line][column];
    }
  }
  for (size_t line = 0; line < N; ++line) {
    copy[line][line + N] = Field(1);
  }

  Gaussian::straight(copy);
  if (copy[N - 1][N - 1] == Field(0)) {
    throw std::invalid_argument("Matrix cannot be inverted.");
  }
  Gaussian::revert(copy);

  for (size_t line = 0; line < N; ++line) {
    for (size_t column = 0; column < N; ++column) {
      data[line][column] = copy[line][column + N];
    }
  }
}

template <size_t N, size_t M, typename Field>
Matrix<N, M, Field> Matrix<N, M, Field>::inverted() const {
  Matrix<N, M, Field> result = *this;
  result.invert();
  return result;
}

template <size_t N, size_t M, typename Field>
Matrix<M, N, Field> Matrix<N, M, Field>::transposed() const {
  Matrix<M, N, Field> result;
  for (size_t line = 0; line < N; ++line) {
    for (size_t column = 0; column < M; ++column) {
      result[column][line] = data[line][column];
    }
  }
  return result;
}

template <size_t N, size_t M, typename Field>
Field Matrix<N, M, Field>::det() const {
  static_assert(N == M);
  update(UpdateCond::NotInitialized);
  return detVal;
}

template <size_t N, size_t M, typename Field>
size_t Matrix<N, M, Field>::rank() const {
  update(UpdateCond::NotInitialized);
  return rankVal;
}

template <size_t N, size_t M, typename Field>
Field Matrix<N, M, Field>::trace() const {
  static_assert(N == M);
  Field result = Field(0);
  for (size_t index = 0; index < N; ++index) {
    result += data[index][index];
  }
  return result;
}

template <size_t N, size_t M, typename Field>
std::array<Field, M> Matrix<N, M, Field>::getRow(size_t line) const {
  return data[line];
}

template <size_t N, size_t M, typename Field>
std::array<Field, N> Matrix<N, M, Field>::getColumn(size_t column) const {
  std::array<Field, N> result;
  for (size_t line = 0; line < N; ++line) {
    result[line] = data[line][column];
  }
  return result;
}

template <size_t N, size_t M, typename Field>
void Matrix<N, M, Field>::update(UpdateCond condition) const {
  if (isInitialized != condition) {
    return;
  }

  Matrix<N, M, Field> copy = *this;
  Gaussian::straight(copy);

  rankVal = N;
  while (rankVal > 0 && copy[rankVal - 1][M - 1] == Field(0)) {
    --rankVal;
  }

  if (N != M) {
    return;
  }

  if (rankVal < N) {
    detVal = Field(0);
  } else {
    detVal = Field(1);
    for (size_t index = 0; index < N; ++index) {
      detVal *= copy.data[index][index];
    }
  }
}

template <size_t N, size_t M, typename Field>
std::istream& operator>>(std::istream& inp, Matrix<N, M, Field>& matrix) {
  for (size_t line = 0; line < N; ++line) {
    for (size_t column = 0; column < M; ++column) {
      inp >> matrix[line][column];
    }
  }
  return inp;
}

template <size_t N, size_t M, typename Field>
std::ostream& operator<<(std::ostream& out, const Matrix<N, M, Field>& matrix) {
  for (size_t line = 0; line < N; ++line) {
    for (size_t column = 0; column < M; ++column) {
      out << matrix[line][column] << ' ';
    }
    out << '\n';
  }
  return out;
}
