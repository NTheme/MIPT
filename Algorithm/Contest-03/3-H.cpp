/*
H (2 балла, с ревью). Вторая статистика (RMQ)

Ограничение времени	1 секунда
Ограничение памяти	64Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Дано число N и последовательность из N целых чисел. Найти вторую порядковую
статистику на заданных диапазонах. Для решения задачи используйте структуру
данных Sparse Table (остальное не засчитывается). Требуемое время обработки
каждого диапазона O(1). Время подготовки структуры данных O(n*log(n)).

Формат ввода
В первой строке заданы 2 числа: размер последовательности N и количество
диапазонов M. Следующие N целых чисел задают последовательность. Далее вводятся
M пар чисел - границ диапазонов.

Формат вывода
Для каждого из M диапазонов напечатать элемент последовательности - 2ю
порядковую статистику. По одному числу в строке.
*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <cmath>
#include <iostream>
#include <vector>

template <typename Type>
std::istream& operator>>(std::istream& inp, std::vector<Type>& array) {
  for (auto& p : array) {
    inp >> p;
  }
  return inp;
}
template <typename Type>
std::ostream& operator<<(std::ostream& out, const std::vector<Type>& array) {
  for (const auto& p : array) {
    out << p << ' ';
  }
  return out;
}

template <typename Type>
class SparseTable {
 public:
  SparseTable() {}
  SparseTable(const std::vector<Type>& array);

  void Build(const std::vector<Type>& array);
  Type Get(size_t left, size_t right);

 private:
  struct FirstValue;

  std::vector<std::vector<std::pair<FirstValue, Type>>> table_;
  std::vector<size_t> logarithm_;

  auto Merge(const std::pair<FirstValue, Type>& left, const std::pair<FirstValue, Type>& right);
};

template <typename Type>
struct SparseTable<Type>::FirstValue {
  Type value;
  size_t index;

  FirstValue();
  FirstValue(Type new_val, size_t new_ind);
  FirstValue& operator=(const FirstValue& right);
};

template <typename Type>
SparseTable<Type>::FirstValue::FirstValue() : value(0), index(0) {}

template <typename Type>
SparseTable<Type>::FirstValue::FirstValue(Type new_val, size_t new_ind) : value(new_val), index(new_ind) {}

template <typename Type>
typename SparseTable<Type>::FirstValue& SparseTable<Type>::FirstValue::operator=(const FirstValue& right) {
  value = right.value;
  index = right.index;
  return *this;
}

template <typename Type>
SparseTable<Type>::SparseTable(const std::vector<Type>& array) {
  Build(array);
}

template <typename Type>
void SparseTable<Type>::Build(const std::vector<Type>& array) {
  table_.clear();
  logarithm_.clear();
  table_.resize(std::log2l(array.size()) + 1);

  logarithm_.push_back(0);
  for (size_t index = 0; index < array.size(); ++index) {
    logarithm_.push_back(logarithm_[index >> 1] + 1);
    table_[0].push_back({{array[index], index}, static_cast<size_t>(1e9)});
  }

  for (size_t power = 1, size = 2; power < table_.size(); ++power, size *= 2) {
    for (size_t index = 0; index + size <= array.size(); ++index) {
      table_[power].push_back(Merge(table_[power - 1][index], table_[power - 1][index + size / 2]));
    }
  }
}

template <typename Type>
Type SparseTable<Type>::Get(size_t left, size_t right) {
  size_t block = logarithm_[right - left - 1], shift = (1 << block);
  return Merge(table_[block][left], table_[block][right - shift]).second;
}

template <typename Type>
auto SparseTable<Type>::Merge(const std::pair<FirstValue, Type>& left, const std::pair<FirstValue, Type>& right) {
  if (left.first.index == right.first.index) {
    return std::pair<FirstValue, Type>(left.first, std::min(left.second, right.second));
  }
  FirstValue first = (left.first.value < right.first.value) ? left.first : right.first;
  Type second = std::min(std::max(left.first.value, right.first.value), std::min(left.second, right.second));
  return std::pair<FirstValue, Type>(first, second);
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t n = 0, m = 0;
  std::cin >> n >> m;

  std::vector<long long> array(n);
  std::cin >> array;

  SparseTable<long long> table(array);
  while (m-- > 0) {
    size_t left = 0, right = 0;
    std::cin >> left >> right;
    std::cout << table.Get(left - 1, right) << '\n';
  }

  return 0;
}
