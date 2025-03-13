#pragma once

#include <algorithm>
#include <cstring>
#include <iostream>

class String {
 public:
  String();
  String(size_t new_size, char symbol);
  String(const char* new_str);
  String(const String& new_str);
  ~String();

  String& operator=(String new_string);
  String& operator+=(const String& new_string);
  String& operator+=(char new_symbol);

  char& operator[](size_t index);
  const char& operator[](size_t index) const;

  size_t size() const;
  size_t length() const;
  size_t capacity() const;
  bool empty() const;

  void push_back(char symbol);
  void pop_back();

  char& front();
  const char& front() const;
  char& back();
  const char& back() const;

  size_t find(const String& substring) const;
  size_t rfind(const String& substring) const;

  String substr(size_t first, size_t length) const;

  void clear();
  void shrink_to_fit();
  void swap(String& other);

  char* data();
  const char* data() const;

 private:
  size_t size_;
  size_t capacity_;
  char* string_;

  void add_endline();
  void realloc(size_t new_capacity);
};

String::String() : size_(0), capacity_(0), string_(new char[1]{'\0'}) {}

String::String(size_t new_size, char symbol)
    : size_(new_size), capacity_(2 * size_), string_(new char[capacity_ + 1]) {
  std::fill(string_, string_ + size_, symbol);
  add_endline();
}

String::String(const char* new_str)
    : size_(std::strlen(new_str)),
      capacity_(2 * size_),
      string_(new char[capacity_ + 1]) {
  std::copy(new_str, new_str + size_ + 1, string_);
}

String::String(const String& new_str)
    : size_(new_str.size_),
      capacity_(new_str.capacity_),
      string_(new char[capacity_ + 1]) {
  std::copy(new_str.string_, new_str.string_ + size_ + 1, string_);
}

String::~String() { delete[] string_; }

String& String::operator=(String new_string) {
  if (string_ != new_string.string_) {
    swap(new_string);
  }
  return *this;
}

String& String::operator+=(const String& new_string) {
  if (size_ + new_string.size_ > capacity_) {
    realloc(2 * (size_ + new_string.size_));
  }
  std::copy(new_string.string_, new_string.string_ + new_string.size_ + 1,
            string_ + size_);

  size_ += new_string.size_;
  return *this;
}

String& String::operator+=(char new_symbol) {
  push_back(new_symbol);
  return *this;
}

String operator+(String left, const String& right) { return left += right; }

String operator+(String left, char right) { return left += right; }

String operator+(char left, const String& right) {
  return String(1, left) += right;
}

bool operator==(const String& left, const String& right) {
  return left.size() == right.size() &&
         std::strcmp(left.data(), right.data()) == 0;
}

bool operator!=(const String& left, const String& right) {
  return !(left == right);
}

bool operator<(const String& left, const String& right) {
  return std::strcmp(left.data(), right.data()) < 0;
}

bool operator>(const String& left, const String& right) { return right < left; }

bool operator>=(const String& left, const String& right) {
  return !(left < right);
}

bool operator<=(const String& left, const String& right) {
  return !(left > right);
}

char& String::operator[](size_t index) { return string_[index]; }

const char& String::operator[](size_t index) const { return string_[index]; }

std::ostream& operator<<(std::ostream& out, const String& string) {
  return out << string.data();
}

std::istream& operator>>(std::istream& inp, String& string) {
  string.clear();

  char buffer;
  while (inp) {
    if (!inp.get(buffer) || (std::isspace(buffer) != 0 && !string.empty())) {
      break;
    }
    if (std::isspace(buffer) == 0) {
      string.push_back(buffer);
    }
  }

  return inp;
}

size_t String::size() const { return size_; }

size_t String::length() const { return size_; }

size_t String::capacity() const { return capacity_; }

bool String::empty() const { return size_ == 0; }

void String::push_back(char symbol) {
  if (size_ + 1 > capacity_) {
    realloc(2 * (size_ + 1));
  }
  ++size_;
  string_[size_ - 1] = symbol;
  add_endline();
}

void String::pop_back() { string_[--size_] = '\0'; }

char& String::front() { return string_[0]; }

const char& String::front() const { return string_[0]; }

char& String::back() { return string_[size_ - 1]; }

const char& String::back() const { return string_[size_ - 1]; }

size_t String::find(const String& substring) const {
  for (size_t index = 0; index + substring.size_ <= size_; ++index) {
    if (std::memcmp(substring.string_, string_ + index, substring.size_) == 0) {
      return index;
    }
  }
  return size_;
}

size_t String::rfind(const String& substring) const {
  for (size_t index = size_ - substring.size_ + 1; index > 0; --index) {
    if (std::memcmp(substring.string_, string_ + index - 1, substring.size_) ==
        0) {
      return index - 1;
    }
  }
  return size_;
}

String String::substr(size_t first, size_t length) const {
  String substring(std::min(length, size_ - first), '\0');
  std::copy(string_ + first, string_ + first + substring.size_,
            substring.string_);
  return substring;
}

void String::clear() {
  size_ = 0;
  add_endline();
}

void String::shrink_to_fit() { realloc(size_); }

void String::swap(String& other) {
  std::swap(size_, other.size_);
  std::swap(capacity_, other.capacity_);
  std::swap(string_, other.string_);
}

char* String::data() { return string_; }

const char* String::data() const { return string_; }

void String::add_endline() { string_[size_] = '\0'; }

void String::realloc(size_t new_capacity) {
  char* new_ptr = new char[new_capacity + 1];
  std::copy(string_, string_ + size_ + 1, new_ptr);
  delete[] string_;
  string_ = new_ptr;
  capacity_ = new_capacity;
}
