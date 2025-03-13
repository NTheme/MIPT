/*
F (3 балла). Звезды

Ограничение времени	1 секунда
Ограничение памяти	64Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

В этой задаче используйте дерево Фенвика.

Марк любит наблюдать за звездами над Пантеллерией. Но следить за ними и над
Риадом, и над Мединой ему крайне тяжело. Поэтому он наблюдает только за частью
пространства, ограниченной кубом размером n × n × n. Этот куб поделен на
маленькие кубики размером 1 × 1 × 1. Во время его наблюдений могут происходить
следующие события:
В каком-то кубике появляются или исчезают несколько звезд.
К нему может заглянуть его друг Рома и поинтересоваться, сколько видно звезд в
части пространства, состоящей из нескольких кубиков.

Формат ввода
Первая строка входного файла содержит натуральное число 1 ≤ n ≤ 128. Координаты
кубиков — целые числа от 0 до n − 1. Далее следуют записи о происходивших
событиях по одной в строке. В начале строки записано число m. Если m равно: 1,
то за ним следуют 4 числа — x, y, z (0 ≤ x, y, z < N) и k (−20000 ≤ k ≤ 20000) —
координаты кубика и величина, на которую в нем изменилось количество видимых
звезд;
2, то за ним следуют 6 чисел — x1, y1, z1, x2, y2, z2 (0 ≤ x1 ≤ x2 < N, 0 ≤ y1 ≤
y2 < N, 0 ≤ z1 ≤ z2 < N), которые означают, что Рома попросил подсчитать
количество звезд в кубиках (x, y, z) из области: x1 ≤ x ≤ x2, y1 ≤ y ≤ y2, z1 ≤
z ≤ z2; 3, то это означает, что Марку надоело наблюдать за звездами и отвечать
на вопросы Ромы. Эта запись встречается во входном файле только один раз и будет
последней записью. Количество записей во входном файле не больше 100002.

Формат вывода
Для каждого вопроса Ромы выведите на отдельной строке одно число — искомое
количество звезд.
*/
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <algorithm>
#include <iostream>
#include <vector>

template <typename TypeLeft, typename TypeRight>
std::istream& operator>>(std::istream& inp,
                         std::pair<TypeLeft, TypeRight>& pair) {
  inp >> pair.first >> pair.second;
  return inp;
}
template <typename TypeLeft, typename TypeRight>
std::ostream& operator<<(std::ostream& out,
                         const std::pair<TypeLeft, TypeRight>& pair) {
  out << pair.first << ' ' << pair.second;
  return out;
}

template <typename Type>
std::istream& operator>>(std::istream& inp, std::vector<Type>& array) {
  for (auto& p : array) {
    inp >> p;
  }
  return inp;
}
template <typename Type>
std::ostream& operator<<(std::ostream& out, const std::vector<Type>& array) {
  for (const auto& p : array) {
    out << p << ' ';
  }
  return out;
}

struct Point {
  int x, y, z;

  Point() : x(0), y(0), z(0) {}
  Point(int x_new, int y_new, int z_new) : x(x_new), y(y_new), z(z_new) {}
};

std::istream& operator>>(std::istream& inp, Point& point) {
  inp >> point.x >> point.y >> point.z;
  return inp;
}

template <typename Type>
class FenwickTree {
 public:
  FenwickTree() {}
  FenwickTree(size_t size) : tree_(size) {}

  Type Get(Point vertex1, Point vertex2);
  void Update(Point point, Type add);

 private:
  class SpaceDim3;

  SpaceDim3 tree_;

  Type GetInternal(Point point);
};

template <typename Type>
class FenwickTree<Type>::SpaceDim3 {
 public:
  SpaceDim3() {}
  SpaceDim3(size_t size)
      : coordinate_(size, std::vector<std::vector<Type>>(
                              size, std::vector<Type>(size + 1))){};

  Type& At(Point point);

  int Size() const;

 private:
  std::vector<std::vector<std::vector<Type>>> coordinate_;
};

template <typename Type>
Type& FenwickTree<Type>::SpaceDim3::At(Point point) {
  if (point.x < 0 || point.y < 0 || point.z < 0) {
    return coordinate_[Size() - 1][Size() - 1][Size() - 1];
  }
  return coordinate_[point.x][point.y][point.z];
}

template <typename Type>
int FenwickTree<Type>::SpaceDim3::Size() const {
  return (int)coordinate_.size();
}

template <typename Type>
Type FenwickTree<Type>::GetInternal(Point point) {
  Type sum = 0;

  for (int x = point.x; x >= 0; x = (x & (x + 1)) - 1) {
    for (int y = point.y; y >= 0; y = (y & (y + 1)) - 1) {
      for (int z = point.z; z >= 0; z = (z & (z + 1)) - 1) {
        sum += tree_.At(Point(x, y, z));
      }
    }
  }
  return sum;
}

template <typename Type>
Type FenwickTree<Type>::Get(Point vertex1, Point vertex2) {
  --vertex1.x, --vertex1.y, --vertex1.z;
  return GetInternal(vertex2) - GetInternal(vertex1) +
         GetInternal({vertex1.x, vertex1.y, vertex2.z}) +
         GetInternal({vertex1.x, vertex2.y, vertex1.z}) +
         GetInternal({vertex2.x, vertex1.y, vertex1.z}) -
         GetInternal({vertex2.x, vertex2.y, vertex1.z}) -
         GetInternal({vertex2.x, vertex1.y, vertex2.z}) -
         GetInternal({vertex1.x, vertex2.y, vertex2.z});
}

template <typename Type>
void FenwickTree<Type>::Update(Point point, Type add) {
  for (int x = point.x; x < tree_.Size(); x = (x | (x + 1))) {
    for (int y = point.y; y < tree_.Size(); y = (y | (y + 1))) {
      for (int z = point.z; z < tree_.Size(); z = (z | (z + 1))) {
        tree_.At(Point(x, y, z)) += add;
      }
    }
  }
}

void ProcessQueries(size_t n) {
  FenwickTree<long long> tree(n);

  size_t query_type = 0;
  while (query_type != 3) {
    std::cin >> query_type;

    if (query_type == 1) {
      Point point;
      long long k;
      std::cin >> point >> k;
      tree.Update(point, k);
    } else if (query_type == 2) {
      Point vertex1, vertex2;
      std::cin >> vertex1 >> vertex2;
      std::cout << tree.Get(vertex1, vertex2) << '\n';
    }
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t n;
  std::cin >> n;

  ProcessQueries(n);

  return 0;
}
