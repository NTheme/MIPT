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

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")
*/

#include <algorithm>
#include <iostream>
#include <unordered_map>
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

struct Point {
  long long x, y;

  Point() : x(0), y(0) {}
  Point(long long x_new, long long y_new) : x(x_new), y(y_new) {}
};

std::istream& operator>>(std::istream& inp, Point& point) {
  inp >> point.x >> point.y;
  return inp;
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t n, t;
  std::cin >> n >> t;

  std::vector<Point> points(n);
  std::cin >> points;

  if (t == 1) {
    size_t ans = 0;
    for (auto& p : points) {
      std::unordered_map<long long, size_t> distance;
      for (auto& q : points) {
        long long dst = (q.x - p.x) * (q.x - p.x) + (q.y - p.y) * (q.y - p.y);
        ++distance[dst];
      }

      for (auto& q : distance) {
        if (q.second > 1) {
          ans += q.second * (q.second - 1) / 2;
        }
      }
    }
    std::cout << ans << '\n';
  } else if (t == 2) {
    std::cout << 0 << '\n';
  }

  return 0;
}
