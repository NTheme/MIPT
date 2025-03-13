/*
C (3 балла). Дефект массы зелья

Ограничение времени	0.8 секунд
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Принц-Полукровка оставил в своем учебнике по зельеварению огромное число
подсказок и заметок. Одна из заметок содержала "Закон несохранения массы". Далее
идет ее текст. Посмотрим на веса каждого из ингредиентов в рецепте и каждому из
них сопоставим столбик единичной ширины и высоты, равной числу граммов
соответствующего ингредиента. Выровняем их снизу по одной линии и получим
"гистограмму ингредиентов". Тогда масса полученного зелья будет равняться
площади самого большого прямоугольника в гистограмме, одна из сторон которого
лежит на общей нижней линии.

Формат ввода
В первой строке входного файла записано число N (0 < N ≤ 10^6) — количество
ингредиентов зелья. Затем следует N целых чисел h1 … hn, где 0 ≤ hi ≤ 10^9. Эти
числа обозначают веса каждого из ингредиентов в граммах.

Формат вывода
Выведите площадь самого большого прямоугольника в гистограмме. Помните, что этот
прямоугольник должен быть на общей базовой линии.
*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <algorithm>
#include <iostream>
#include <stack>

void FindLess(int n, const long long* height, long long** next_less) {
  std::stack<long long> st[2];
  st[0].push(0), st[1].push(0);

  for (int i = 1; i < n + 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      int index = j * (n + 2 - 2 * i) + i;
      while (height[st[j].top()] > height[index]) {
        next_less[j][st[j].top()] = index;
        st[j].pop();
      }

      st[j].push(index);
    }
  }
}

long long CountMaxSquare(const long long* height, int n) {
  long long* next_less[2] = {new long long[n + 2], new long long[n + 2]};
  for (int i = 0; i < n + 2; ++i) {
    next_less[0][i] = next_less[1][i] = 0;
  }

  FindLess(n, height, next_less);

  long long square = 0;
  for (int i = 1; i < n + 1; ++i) {
    square = std::max(
        square, height[i] * std::abs(next_less[0][i] - next_less[1][i] - 1));
  }

  delete[] next_less[0];
  delete[] next_less[1];

  return square;
}

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(0), std::cout.tie(0);

  int n = 0;
  std::cin >> n;

  long long* height = new long long[n + 2];
  height[0] = height[n + 1] = -2e9 - 1;

  for (int i = 1; i < n + 1; ++i) {
    std::cin >> height[i];
  }

  std::cout << CountMaxSquare(height, n) << '\n';

  delete[] height;

  return 0;
}
