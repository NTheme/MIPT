/*
I (2 балла). Минимальный люмос

Ограничение времени	3 секунды
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Трио Рон, Гарри и Гермиона попало в очередную западню. Для открытия двери им
нужно осветить все N приемников, стоящих в ряд. При этом заклятие «Люмос»
способно покрыть светом отрезок длины ℓ. Хотя силы данной троицы и велики, они
не могут поддерживать одновременно более k активных заклинаний «Люмос». Найдите
минимальное ℓ, чтобы трио могло выбраться.

Формат ввода
На первой строке n (1 ≤ n ≤ 10^5) и k (1 ≤ k ≤ n).
На второй n чисел xi (∣xi∣ ≤ 10^9) — координаты приемников.

Формат вывода
Минимальное такое ℓ, что точки можно покрыть k отрезками длины ℓ.
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

int Solve(long long* arr, int n, int k) {
  std::sort(arr, arr + n);

  long long l = -1, r = 2e9;
  while (r - l > 1) {
    long long m = (l + r) / 2, pos = 0, cou = 0;
    for (; pos < n; ++cou) {
      pos = std::upper_bound(arr + pos, arr + n, arr[pos] + m) - arr;
    }

    if (cou > k) {
      l = m;
    } else {
      r = m;
    }
  }

  return r;
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(0), std::cout.tie(0);
  std::cout.precision(20);
  std::srand((unsigned int)time(NULL));

  int n = 0, k = 0;
  std::cin >> n >> k;

  long long* arr = new long long[n];

  for (int i = 0; i < n; i++) {
    std::cin >> arr[i];
  }

  std::cout << Solve(arr, n, k);

  delete[] arr;
  return 0;
}
