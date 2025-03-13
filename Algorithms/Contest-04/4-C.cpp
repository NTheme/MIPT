/*

C (3 балла). Команда R и сдвиг

Ограничение времени	1 секунда
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Команда R вновь стремится выкрасть Пикачу! Для их коварного плана они придумали
устройство, способное генерировать исключительно правдоподобные голограммы
покемонов. После этого Джесси и Джеймс сделали множество голограмм покемонов, но
с малыми видоизменениями в силу погрешности проектора. У Хэша, владельца Пикачу,
есть специальное устройство, которое по наведению на покемона умеет выдавать
основную информацию о нем, — покидекс. В силу особенности данного прибора,
голограммы называются словами, похожими на оригинальные виды. Слова называют
похожими, если из одного слова можно получить другое, заменив в нём одинаковые
буквы на одинаковые, а разные — на разные. Например, слова «aba» и «bcb»
похожие, а «abb» и «aaa» — нет. В ходе битвы с командой R Хэшу придется активно
пользоваться покидексом, чтобы понимать, где реальные цели, а где их копии. Вам
предстоит написать часть покидекса, отвечающую за запоминание встреченных в ходе
битвы покемонов. Всего будет q запросов. Запрос на добавление слова s_i
означает, что Хэш хочет добавить этого покемона в память покидекса. Запрос на
проверку покемона с именем s_i означает, что Хэш хочет узнать, есть ли в его
сканере покемонов, чье название похоже на s_i. В частности он поймет, что перед
ним голограмма, если такой уже есть в памяти. Напишите программу, выполняющую
все запросы Хэша.

Формат ввода
В первой строке содержится число q (1 ≤ q ≤ 10^5) — количество запросов. В
следующих q строках содержатся сами запросы. Запрос на добавление в начале
содержит символ «+», а запрос на проверку в начале содержит символ «?». Затем
идёт само слово, которое необходимо добавить в покидекс или проверить,
содержатся ли в памяти покидекса похожие на него слова. Обозначим через L
суммарную длину слов в запросах. Гарантируется, что L не превосходит 10^6.

Формат вывода
На каждый запрос проверки выведите в отдельной строке «YES» (без кавычек), если
похожее слово есть в словаре, и «NO» (без кавычек) в противном случае.

*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <algorithm>
#include <iostream>
#include <vector>

class HashTable {
 public:
  HashTable(size_t num_hashes, size_t capacity)
      : table_(num_hashes, std::vector<bool>(capacity)) {}
  void Add(const std::vector<size_t>& elem);
  bool Exist(const std::vector<size_t>& elem);

 private:
  std::vector<std::vector<bool>> table_;
};

void HashTable::Add(const std::vector<size_t>& elem) {
  for (size_t index = 0; index < elem.size(); ++index) {
    table_[index][elem[index]] = true;
  }
}

bool HashTable::Exist(const std::vector<size_t>& elem) {
  for (size_t index = 0; index < elem.size(); ++index) {
    if (!table_[index][elem[index]]) {
      return false;
    }
  }

  return true;
}

size_t CountHash(const std::vector<size_t>& arr, const size_t kMul,
                 const size_t kMod) {
  size_t hash = 0;
  for (const auto& p : arr) {
    hash = (hash * kMul + p) % kMod;
  }
  return hash;
}

std::vector<size_t> GetHash(const std::string& elem, const size_t kMod) {
  std::vector<size_t> hash;

  size_t num_var = 0;
  std::vector<size_t> element(256);
  std::vector<size_t> recalc(elem.size());
  for (size_t i = 0; i < elem.size(); ++i) {
    if (element[static_cast<size_t>(elem[i])] == 0) {
      element[static_cast<size_t>(elem[i])] = ++num_var;
    }
    recalc[i] = element[static_cast<size_t>(elem[i])];
  }
  hash.push_back(CountHash(recalc, 37, kMod));
  hash.push_back(CountHash(recalc, 19, kMod));
  hash.push_back(CountHash(recalc, 59, kMod));

  return hash;
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t q;
  std::cin >> q;

  const size_t kMod = 1e6;
  const size_t kNum = 3;

  HashTable table(kNum, kMod);
  while (q-- > 0) {
    char word;
    std::string str;
    std::cin >> word >> str;

    auto hash = GetHash(str, kMod);

    if (word == '+') {
      table.Add(hash);
    } else if (word == '?') {
      std::cout << (table.Exist(hash) ? "YES" : "NO") << '\n';
    }
  }

  return 0;
}
