/*

G (3 балла, с ревью). Фыр-Фыр-Фенвик

Ограничение времени	2 секунды
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Даны n точек с весами на плоскости. Каждая задаётся тремя числами x_i, y_i, w_i
(координаты и вес). Вам нужно обработать m запросов двух типов: get rx ry –
посчитать сумму весов точек, у которых x_i ≤ rx и y_i ≤ r_y, change i z – задать
i-й точке новый вес равный z.

Формат ввода
На первой строке число n (1 ≤ n ≤ 100000). На следующих n строках тройки целых
чисел x_i, y_i, w_i (0 ≤ x_i, y_i, w_i < 10^9). Следующая строка содержит
количество запросов m (1 ≤ m ≤ 300000). На следующих m строках описания запросов
в формате get rx, ry и change i z. Здесь 1 ≤ i ≤ n, а остальные числа целые от 0
до 10^9 − 1.

Формат вывода
Для каждого запроса типа “get” выведите одно целое число на отдельной строке —
ответ на запрос.

*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <algorithm>
#include <iostream>
#include <vector>

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

struct WeightedPoint {
  long long x, y;
  long long weight;

  WeightedPoint() : x(0), y(0), weight(0) {}
  WeightedPoint(long long x_new, long long y_new, long long weight_new) : x(x_new), y(y_new), weight(weight_new) {}

  static bool CompareX(const WeightedPoint& left, const WeightedPoint& right);
  static bool CompareY(const WeightedPoint& left, const WeightedPoint& right);
};

bool WeightedPoint::CompareX(const WeightedPoint& left, const WeightedPoint& right) { return left.x < right.x; }

bool WeightedPoint::CompareY(const WeightedPoint& left, const WeightedPoint& right) { return left.y < right.y; }

std::istream& operator>>(std::istream& inp, WeightedPoint& point) {
  inp >> point.x >> point.y >> point.weight;
  return inp;
}

class FenwickTreeY {
 public:
  FenwickTreeY() {}
  FenwickTreeY(const std::vector<WeightedPoint>& points);

  void Build(std::vector<WeightedPoint> points);
  long long Get(long long point) const;
  void Update(long long point, long long add);

 private:
  std::vector<long long> compressed_;
  std::vector<long long> tree_;
};

FenwickTreeY::FenwickTreeY(const std::vector<WeightedPoint>& points) { Build(points); }

void FenwickTreeY::Build(std::vector<WeightedPoint> points) {
  std::sort(points.begin(), points.end(), WeightedPoint::CompareY);

  compressed_.assign(points.size(), 0);
  for (size_t index = 0; index < points.size(); ++index) {
    compressed_[index] = points[index].y;
  }
  compressed_.resize(std::unique(compressed_.begin(), compressed_.end()) - compressed_.begin());
  tree_.assign(compressed_.size() + 1, 0);

  for (size_t index = 0; index < points.size(); ++index) {
    Update(points[index].y, points[index].weight);
  }
}

long long FenwickTreeY::Get(long long point) const {
  auto point_index = std::upper_bound(compressed_.begin(), compressed_.end(), point);
  size_t index = std::distance(compressed_.begin(), point_index);

  long long sum = 0;
  for (; index > 0; index &= (index - 1)) {
    sum += tree_[index];
  }
  return sum;
}

void FenwickTreeY::Update(long long point, long long add) {
  auto point_index = std::lower_bound(compressed_.begin(), compressed_.end(), point);
  size_t index = std::distance(compressed_.begin(), point_index) + 1;

  for (; index < tree_.size(); index = (index | (index - 1)) + 1) {
    tree_[index] += add;
  }
}

class FenwickTreeX {
 public:
  FenwickTreeX() {}
  FenwickTreeX(const std::vector<WeightedPoint>& points);

  void Build(std::vector<WeightedPoint> points);
  long long Get(WeightedPoint point) const;
  void Update(WeightedPoint point);

 private:
  std::vector<long long> compressed_;
  std::vector<FenwickTreeY> tree_;
};

FenwickTreeX::FenwickTreeX(const std::vector<WeightedPoint>& points) { Build(points); }

void FenwickTreeX::Build(std::vector<WeightedPoint> points) {
  std::sort(points.begin(), points.end(), WeightedPoint::CompareX);

  compressed_.assign(points.size(), 0);
  for (size_t index = 0; index < points.size(); ++index) {
    compressed_[index] = points[index].x;
  }
  compressed_.resize(std::unique(compressed_.begin(), compressed_.end()) - compressed_.begin());
  tree_.resize(compressed_.size() + 1);

  for (size_t index = 1; index < tree_.size(); ++index) {
    auto left = std::lower_bound(points.begin(), points.end(), WeightedPoint(compressed_[(index & (index - 1))], 0, 0),
                                 WeightedPoint::CompareX);
    auto right = std::upper_bound(points.begin(), points.end(), WeightedPoint(compressed_[index - 1], 0, 0),
                                  WeightedPoint::CompareX);
    std::vector<WeightedPoint> tree_y;
    for (auto ptr = left; ptr < right; ++ptr) {
      tree_y.push_back(*ptr);
    }
    tree_[index].Build(tree_y);
  }
}

long long FenwickTreeX::Get(WeightedPoint point) const {
  auto point_index = std::upper_bound(compressed_.begin(), compressed_.end(), point.x);
  size_t index = std::distance(compressed_.begin(), point_index);

  long long sum = 0;
  for (; index > 0; index &= (index - 1)) {
    sum += tree_[index].Get(point.y);
  }
  return sum;
}

void FenwickTreeX::Update(WeightedPoint point) {
  auto point_index = std::lower_bound(compressed_.begin(), compressed_.end(), point.x);
  size_t index = std::distance(compressed_.begin(), point_index) + 1;

  for (; index < tree_.size(); index = (index | (index - 1)) + 1) {
    tree_[index].Update(point.y, point.weight);
  }
}

class WeightedPoints {
 public:
  WeightedPoints(const std::vector<WeightedPoint>& points_new) : points_(points_new), tree_(points_) {}

  void Update(size_t index, long long weight);
  long long Get(const WeightedPoint& point) const;

 private:
  std::vector<WeightedPoint> points_;
  FenwickTreeX tree_;
};

void WeightedPoints::Update(size_t index, long long weight) {
  tree_.Update({points_[index - 1].x, points_[index - 1].y, weight - points_[index - 1].weight});
  points_[index - 1].weight = weight;
}

long long WeightedPoints::Get(const WeightedPoint& point) const { return tree_.Get(point); }

void ProcessQueries(std::vector<WeightedPoint>& points_input) {
  WeightedPoints points(points_input);
  size_t m;
  std::cin >> m;

  while (m-- > 0) {
    std::string query_type;
    std::cin >> query_type;

    if (query_type == "change") {
      size_t index;
      long long weight_new;
      std::cin >> index >> weight_new;
      points.Update(index, weight_new);
    } else if (query_type == "get") {
      WeightedPoint point;
      std::cin >> point.x >> point.y;
      std::cout << points.Get(point) << '\n';
    }
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t n;
  std::cin >> n;

  std::vector<WeightedPoint> points_input(n);
  std::cin >> points_input;

  ProcessQueries(points_input);

  return 0;
}
