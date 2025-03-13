/*
F (1 балл). Нимбус минус две тысячи

Ограничение времени	1 секунда
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Очередная игра в квиддич. Когтевранцы выбирают, кто будет представлять их
факультет на столь важном событии. Всего есть N кандидатов, у каждого метла
летает с максимальной скоростью ai. В силу определенных правил формирования
команды, можно выбрать обладателей метел с номерами 1, 2, … , l,  r, r + 1, … ,
N. Когтевранцы уверены, что скорость команды определяется как скорость самой
медленной метлы в команде. Формат ввода В первой строке идет число N (1 ≤ N ≤
10^5) — число кандидатов. Во второй строке идут N чисел ai (−10^9 ≤ ai ≤ 10^9) —
скорости метел (не спрашивайте, что такое отрицательная скорость, вы же всего
лишь маглы, вам не понять). Далее на отдельной строке идет число Q (1 ≤ Q ≤
10^6) — число запросов скорости команды от капитана. Вход завершается Q строками
по два числа li, ri (1 ≤ li ≤ ri ≤ N) — границы i-го запроса.

Формат вывода
Выведите Q строк — ответы на запросы.
*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <algorithm>
#include <iostream>

template <typename T1, typename T2>
std::istream& operator>>(std::istream& inp, std::pair<T1, T2>& a) {
  inp >> a.first >> a.second;
  return inp;
}
template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& out, const std::pair<T1, T2>& a) {
  out << a.first << ' ' << a.second;
  return out;
}

void MakeMin(int n, int* speed, int* minl, int* minr) {
  minl[0] = speed[0];
  for (int i = 1; i < n; ++i) {
    minl[i] = std::min(speed[i], minl[i - 1]);
  }

  minr[n - 1] = speed[n - 1];
  for (int i = n - 2; i >= 0; --i) {
    minr[i] = std::min(speed[i], minr[i + 1]);
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(0), std::cout.tie(0);
  std::cout.precision(20);

  int n = 0;
  std::cin >> n;

  int* speed = new int[n];
  for (int i = 0; i < n; ++i) {
    std::cin >> speed[i];
  }

  int* minl = new int[n];
  int* minr = new int[n];
  MakeMin(n, speed, minl, minr);

  int q = 0;
  std::cin >> q;

  while (q-- > 0) {
    int l = 0, r = 0;
    std::cin >> l >> r;
    std::cout << std::min(minl[l - 1], minr[r - 1]) << '\n';
  }

  delete[] minl;
  delete[] minr;
  delete[] speed;
  return 0;
}
