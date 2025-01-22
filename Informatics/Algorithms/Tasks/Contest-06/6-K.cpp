/*
K (1 балл, с ревью). Правдивые поручения
Ограничение времени	1 секунда
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt
Карлу необходимо выполнить для мистера Правды N поручений, каждое из них
характеризуется двумя числами: необходимое число ресурсов m и награда c. Сиджею
негде набирать ресурсы, так что он ограничен M единицами ресурсов. Какие задания
он может выполнить, чтобы максимизировать награду?
Формат ввода
В первой строке вводится натуральное число N, не превышающее 100 и натуральное
число M, не превышающее 10000.
Во второй строке вводятся N натуральных чисел mi, не превышающих 100.
Во третьей строке вводятся N натуральных чисел сi, не превышающих 100.
Формат вывода
Выведите номера поручений (числа от 1 до N), которые войдут в оптимальный набор.
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

std::vector<size_t> GetOptimalSequence(const std::vector<int>& cost,
                                       const std::vector<int>& reward,
                                       size_t resources) {
  std::vector<std::vector<int>> dp(cost.size() + 1,
                                   std::vector<int>(resources + 1, -1));
  std::vector<std::vector<size_t>> prev(cost.size() + 1,
                                        std::vector<size_t>(resources + 1));

  dp[0][resources] = 0;
  for (size_t last = 1; last <= cost.size(); ++last) {
    for (size_t rest = 0; rest <= resources; ++rest) {
      if (dp[last - 1][rest] != -1) {
        dp[last][rest] = dp[last - 1][rest];
        prev[last][rest] = prev[last - 1][rest];
      }

      size_t prev_rest = rest + cost[last - 1];
      if (prev_rest <= resources && dp[last - 1][prev_rest] != -1 &&
          dp[last][rest] < dp[last - 1][prev_rest] + reward[last - 1]) {
        dp[last][rest] = dp[last - 1][prev_rest] + reward[last - 1];
        prev[last][rest] = last;
      }
    }
  }

  std::vector<size_t> sequence;
  int optimal_rest = 0;
  for (size_t rest = 0; rest < resources; ++rest) {
    if (dp[cost.size()][optimal_rest] < dp[cost.size()][rest]) {
      optimal_rest = rest;
    }
  }

  size_t position = cost.size(), current = prev[position][optimal_rest];
  while (current != 0) {
    sequence.push_back(current);
    position = current - 1;
    optimal_rest += cost[current - 1];
    current = prev[position][optimal_rest];
  }

  std::reverse(sequence.begin(), sequence.end());

  return sequence;
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t n, m;
  std::cin >> n >> m;
  std::vector<int> cost(n), reward(n);
  std::cin >> cost >> reward;

  auto sequence = GetOptimalSequence(cost, reward, m);
  std::cout << sequence;

  return 0;
}
