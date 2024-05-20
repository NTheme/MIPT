/*
P (4 балла, с ревью). Оборона города

Ограничение времени	1 секунда
Ограничение памяти	512Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

В последних миссиях Балласы воспользовались неразберихой в Лос-Сантосе. Сиджей
готовится к обороне своего города. Его город представляет собой клетчатый
прямоугольник размера n × m, в котором каждая клетка — отдельный район. Балласы
могут либо атаковать район, либо пощадить его. При этом есть районы, в которых
достаточно оборонительных сооружений до следующего конца света, и балласы не в
силах их захватить, а есть те, в которых никакой защиты нет, и им в любом случае
придется капитулировать. Балласы все ещё справедливы, а это значит, что в любом
квадрате размера 2 × 2 должно быть поровну до зубов защищенных и безоружных
районов. Теперь балласы хотят узнать количество различных вариантов
распределения районов города на безоружные и излишне защищенные.

Формат ввода
В первой строке входного файла задано два целых числа n и m (1 ≤ n ≤ 15 и 1 ≤ m
≤ 100) — размер города. Далее следует n строк по m символов в каждой, где символ
‘+’ означает, что соответствующий район не может быть захвачен, символ ‘—’ —
соответствующий район будет захвачен в любом случае и символ ‘.’ — Балласы могут
решить, что им делать с этим районом.

Формат вывода
Выведите количество различных вариантов напасть на город по модулю 10^9 + 7
*/

#include <algorithm>
#include <iostream>
#include <vector>

template <typename Type>
std::istream& operator>>(std::istream& inp, std::vector<Type>& array) {
  for (auto& elem : array) {
    inp >> elem;
  }
  return inp;
}
template <typename Type>
std::ostream& operator<<(std::ostream& out, const std::vector<Type>& array) {
  for (const auto& elem : array) {
    out << elem << ' ';
  }
  return out;
}

static const int kMod = 1e9 + 7;

std::vector<std::vector<int>> RetrieveInput(
    const std::vector<std::string>& safe) {
  std::vector<std::vector<int>> field(safe.size(),
                                      std::vector<int>(safe.back().size()));
  for (size_t row = 0; row < field.size(); ++row) {
    for (size_t col = 0; col < field.back().size(); ++col) {
      field[row][col] =
          (safe[row][col] == '+') ? 1 : ((safe[row][col] == '-') ? 0 : -1);
    }
  }
  return field;
}

std::vector<std::vector<int>> CountPrevMask(size_t n) {
  std::vector<std::vector<int>> prev_mask(1 << n, std::vector<int>(2));
  for (int mask = 0; mask < (1 << n); ++mask) {
    for (size_t first = 0; first < 2; ++first) {
      prev_mask[mask][first] = first;
      for (size_t row = 1; row < n; ++row) {
        int num_safe = ((mask >> row) & 1) + ((mask >> (row - 1)) & 1) +
                       ((prev_mask[mask][first] >> (row - 1)) & 1);
        if (num_safe != 1 && num_safe != 2) {
          prev_mask[mask][first] = (1 << n);
          break;
        }
        prev_mask[mask][first] += ((num_safe % 2) << row);
      }
    }
  }
  return prev_mask;
}

bool CheckMask(const std::vector<std::vector<int>>& field, size_t col,
               int mask) {
  for (size_t row = 0; row < field.size(); ++row) {
    if (field[row][col] != -1 && ((mask >> row) & 1) != field[row][col]) {
      return false;
    }
  }
  return true;
}

std::vector<std::vector<int>> CountDP(
    const std::vector<std::vector<int>>& field,
    const std::vector<std::vector<int>>& prev_mask) {
  std::vector<std::vector<int>> dp(1 << field.size(),
                                   std::vector<int>(field.back().size()));
  for (int mask = 0; mask < (1 << field.size()); ++mask) {
    dp[mask][0] = static_cast<int>(CheckMask(field, 0, mask));
    // если маска подходит под уже имеющуюся расстановку, то есть не
    // противоречит данной в условии в 1 столбце, то 1, иначе - 0
    // затем пересчет из квадратика 2 на 2 соседних значений
  }

  for (size_t col = 1; col < field.back().size(); ++col) {
    for (int mask = 0; mask < (1 << field.size()); ++mask) {
      if (!CheckMask(field, col, mask)) {
        continue;
      }

      for (size_t first = 0; first < 2; ++first) {
        if (prev_mask[mask][first] == (1 << field.size()) ||
            !CheckMask(field, col - 1, prev_mask[mask][first])) {
          continue;
        }
        dp[mask][col] =
            (dp[prev_mask[mask][first]][col - 1] + dp[mask][col]) % kMod;
      }
    }
  }

  return dp;
}

int CountVariants(const std::vector<std::string>& safe) {
  auto field = RetrieveInput(safe);
  auto prev_mask = CountPrevMask(safe.size());
  auto dp = CountDP(field, prev_mask);

  int num_of_variants = 0;
  for (int mask = 0; mask < (1 << safe.size()); ++mask) {
    num_of_variants =
        (num_of_variants + dp[mask][safe.back().size() - 1]) % kMod;
  }

  return num_of_variants;
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  int n, m;
  std::cin >> n >> m;
  std::vector<std::string> safe(n);
  std::cin >> safe;
  std::cout << CountVariants(safe) << '\n';

  return 0;
}
