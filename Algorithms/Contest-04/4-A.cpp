/*
A (2 балла, с ревью). Эш-таблица
Ограничение времени	4 секунды
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt
Нужно реализовать структуру данных множество, способную быстро выполнять
следующие операции:
«+ N» — добавить число N в множество. Не гарантируется, что N отсутствует в
множестве.
«- N» — удалить число N из множества. Не гарантируется, что N имеется
в множестве.
«? N» — узнать, есть ли число N в множестве.
Формат ввода
В первой строке идет число N (1 ≤ N ≤ 10^6) — число запросов к множеству. Далее
идет N запросов на N строках в формате выше.
Все числа из запросов лежат в отрезке [0, 10^9]
Формат вывода
Для каждого запроса третьего типа вывести YES, если ответ положителен, и NO —
иначе.
*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <algorithm>
#include <iostream>
#include <vector>

template <typename Type>
class HashTable {
 public:
  explicit HashTable(size_t capacity) : mod_(capacity), table_(mod_) {}
  void Add(Type elem);
  void Remove(Type elem);
  bool Exist(Type elem) const;

 private:
  size_t mod_;
  std::vector<std::vector<Type>> table_;

  size_t GetHash(Type elem) const;
  void Reallocate(size_t new_capacity);
};

template <typename Type>
void HashTable<Type>::Add(Type elem) {
  size_t hash = GetHash(elem);
  auto iterator = std::find(table_[hash].begin(), table_[hash].end(), elem);
  if (iterator == table_[hash].end()) {
    table_[hash].push_back(elem);
  }
}

template <typename Type>
void HashTable<Type>::Remove(Type elem) {
  size_t hash = GetHash(elem);
  auto iterator = std::find(table_[hash].begin(), table_[hash].end(), elem);
  if (iterator != table_[hash].end()) {
    table_[hash].erase(iterator);
  }
}

template <typename Type>
bool HashTable<Type>::Exist(Type elem) const {
  size_t hash = GetHash(elem);
  return std::find(table_[hash].begin(), table_[hash].end(), elem) !=
         table_[hash].end();
}

template <typename Type>
size_t HashTable<Type>::GetHash(Type elem) const {
  return elem % mod_;
}

void ProcessQueries(size_t queries) {
  HashTable<int> table(queries);
  for (size_t iter = 0; iter < queries; ++iter) {
    char type;
    int num;
    std::cin >> type >> num;

    if (type == '+') {
      table.Add(num);
    } else if (type == '-') {
      table.Remove(num);
    } else if (type == '?') {
      std::cout << (table.Exist(num) ? "YES\n" : "NO\n");
    }
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t queries;
  std::cin >> queries;

  ProcessQueries(queries);

  return 0;
}
