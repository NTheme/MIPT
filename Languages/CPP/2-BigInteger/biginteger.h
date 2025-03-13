#pragma once

#include <algorithm> 
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
