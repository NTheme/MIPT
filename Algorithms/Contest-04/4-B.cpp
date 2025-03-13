/*

B (1 балл). Общая коллекция

Ограничение времени	5 секунд
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

У тренеров покемонов огромное множество самых разных зверьков. Хэш Кетчум и
Мисти решили посмотреть, а какие у них общие? Для удобства все покемоны
пронумерованы, найдите общих покемонов с учетом кратности.

Формат ввода
В первой строке дано число N (1 ≤ N ≤ 3 * 10^6) — число покемонов у Хэша. На
следующей строке идут N чисел (1 ≤ a_i ≤ 10^9) — номера покемонов, которые есть
у Хэша. В третьей и четвертой строке в том же формате и с теми же ограничениями
описана коллекция Мисти.

Формат вывода
Выведите пересечение коллекций в формате как в входном файле. Порядок не важен.

*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <algorithm>
#include <iostream>
#include <vector>

template <typename T>
std::istream& operator>>(std::istream& inp, std::vector<T>& arr) {
  for (auto& p : arr) {
    inp >> p;
  }
  return inp;
}
template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& arr) {
  for (const auto& p : arr) {
    out << p << ' ';
  }
  return out;
}

template <typename Type>
class HashTable {
 public:
  HashTable(size_t capacity_new)
      : capacity_(capacity_new), table_(capacity_), num_(capacity_) {}
  void Add(Type elem);
  bool Remove(Type elem);

 private:
  size_t capacity_;
  std::vector<std::vector<Type>> table_;
  std::vector<std::vector<size_t>> num_;

  size_t GetHash(Type elem);
  void Reallocate(size_t new_capacity);
};

template <typename Type>
void HashTable<Type>::Add(Type elem) {
  size_t hash = GetHash(elem);
  size_t index = std::find(table_[hash].begin(), table_[hash].end(), elem) -
                 table_[hash].begin();
  if (index == table_[hash].size()) {
    table_[hash].push_back(elem);
    num_[hash].push_back(1);
  } else {
    ++num_[hash][index];
  }
}

template <typename Type>
bool HashTable<Type>::Remove(Type elem) {
  size_t hash = GetHash(elem);
  size_t index = std::find(table_[hash].begin(), table_[hash].end(), elem) -
                 table_[hash].begin();
  if (index != table_[hash].size()) {
    if (num_[hash][index] == 1) {
      table_[hash].erase(table_[hash].begin() + index);
      num_[hash].erase(num_[hash].begin() + index);
    } else {
      --num_[hash][index];
    }

    return true;
  }

  return false;
}

template <typename Type>
size_t HashTable<Type>::GetHash(Type elem) {
  return elem % capacity_;
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t n;
  std::cin >> n;
  std::vector<int> first(n);
  std::cin >> first;

  size_t m;
  std::cin >> m;
  std::vector<int> second(m);
  std::cin >> second;

  HashTable<int> table(std::max(static_cast<int>(n / 10), 1));
  for (auto& p : first) {
    table.Add(p);
  }

  std::vector<int> intersection;
  for (auto& q : second) {
    if (table.Remove(q)) {
      intersection.push_back(q);
    }
  }

  std::cout << intersection.size() << '\n' << intersection << '\n';

  return 0;
}
