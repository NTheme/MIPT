/*
G (3 балла, с ревью). Средний шум по теплице

Ограничение времени	2 секунды
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Мандрагоры — безопасные растения, но их корни могут издавать оглушающие звуки.
На втором курсе студенты Хогвартса в качестве одного из занятий пересаживают уже
подросших мандрагор из маленьких горшков в горшки побольше. Горшки с
мандрагорами стоят в ряд, поэтому их пронумеровали от 1 до N. Профессор Стебль
рассказала, что каждая из мандрагор кричит с громкостью ai децибел.

Гермиона, как самая одаренная студентка, узнала, что можно за раз пересадить
несколько подряд идущих мандрагор, при этом их усредненный шум будет звучать как
среднее геометрическое их громкостей по отдельности. Так как математика в
Хогвартсе не преподается, вам придется ей помочь.

Формат ввода
В первой строке дано число N (1 ≤ N ≤ 3 ⋅ 10^5) — число мандрагор. На второй
строке идут N вещественных чисел (0.01 ≤ ai ≤ 103) с двумя знаками после
десятичной точки — громкость i-й мандрагоры. На третьей строке идет единственное
число Q (1 ≤ Q ≤ 105) — число запросов подсчета среднего шума на подотрезке от
Гермионы. Далее идет Q строк запросов в формате «i j» (0 ≤ i ≤ j ≤ N - 1): i, j
— индексы массива, задающие отрезок для запроса.

Формат вывода
Для каждого запроса на новой строке выведите полученную среднюю громкость
результат с точностью до не менее шестого знака после запятой.
*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <cmath>
#include <iomanip>
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

void CountPrefixAVG(long double* a, long double* b, int n) {
  b[0] = 1, b[1] = a[0];
  for (int i = 1; i < n; ++i) {
    long double sqr = i + 1;
    b[i + 1] = expl(logl(b[i]) * i / sqr) * expl(logl(a[i]) / sqr);
  }
}

long double CountAVG(long double* b, int i, int j) {
  long double sqr = j - i + 1;
  return expl(logl(b[j + 1]) * (j + 1) / sqr) / expl(logl(b[i]) * i / sqr);
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(0), std::cout.tie(0);
  std::cout.precision(20);
  std::srand((unsigned int)time(NULL));

  int n = 0;
  std::cin >> n;

  long double* a = new long double[n];
  for (int i = 0; i < n; ++i) {
    std::cin >> a[i];
  }

  long double* b = new long double[n + 1];
  CountPrefixAVG(a, b, n);

  int q = 0;
  std::cin >> q;
  while (q-- > 0) {
    int i = 0, j = 0;
    std::cin >> i >> j;
    std::cout << CountAVG(b, i, j) << '\n';
  }

  delete[] a;
  delete[] b;
  return 0;
}