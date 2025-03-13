/*

A (1 балл). Деккардова организация

Ограничение времени	0.2 секунды
Ограничение памяти	16Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Доминик решил пересмотреть свой автопарк и организовать информацию о нем. Для
этого он присвоил каждой машине номер и приоритет. Доминик уверен, что если
расположить машины как вершины в декартовом дерева поиска, то он сможет
подбирать нужную тачку на гонку оптимально.

Так как Доминик, как и всегда, торопится на встречу с семьей, ему нужно
построить дерево для хранения информации о машинах за линейное время от числа
машин.

Формат ввода
В первой строке записано число N — количество пар номер-приоритет. Далее следует
N (1 ≤ N ≤ 50000) пар (ai, bi). Для всех пар |ai|, |bi| ≤ 30000. ai ≠ aj и bi ≠
bj для всех i ≠ j. Гарантируется, что пары отсортированы по возрастанию ai.

Формат вывода
Если Доминик выбрал неверные приоритеты, и дерево построить невозможно, то
выведите NO. Иначе выведите N строк, каждая из которых должна описывать вершину.
Описание вершины состоит из трёх чисел: номер предка, номер левого сына и номер
правого сына. Если у вершины отсутствует предок или какой-либо из сыновей, то
выводите на его месте число 0.

Если подходящих деревьев несколько, выведите любое.

*/

#include <algorithm>
#include <compare>
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

  bool operator==(const Point& other) const;
};

bool Point::operator==(const Point& other) const {
  return x == other.x && y == other.y;
}

std::istream& operator>>(std::istream& inp, Point& point) {
  inp >> point.x >> point.y;
  return inp;
}

class Treap {
 public:
  Treap();
  Treap(const std::vector<Point>& val);
  ~Treap();

  void Build(const std::vector<Point>& val);
  void PrintVertex() const;

 private:
  struct Vertex;

  Vertex* root_;
  size_t size_;

  void PrintRec(
      Vertex* current,
      std::vector<std::pair<size_t, std::pair<size_t, size_t>>>& ver) const;
};

struct Treap::Vertex {
  Point val;
  size_t num;

  Vertex* left;
  Vertex* right;
  Vertex* parent;

  Vertex(Point val_new, size_t num_new);
  ~Vertex();
};

Treap::Vertex::Vertex(Point val_new, size_t num_new)
    : val(val_new),
      num(num_new),
      left(nullptr),
      right(nullptr),
      parent(nullptr) {}

Treap::Vertex::~Vertex() {
  delete left;
  delete right;
}

Treap::Treap() : root_(nullptr), size_(0) {}

Treap::Treap(const std::vector<Point>& val) : root_(nullptr) { Build(val); }

Treap::~Treap() { delete root_; }

void Treap::Build(const std::vector<Point>& val) {
  size_ = val.size();
  if (val.empty()) {
    return;
  }

  delete root_;
  root_ = new Vertex(val[0], 1);
  for (size_t index = 1; index < val.size(); ++index) {
    if (root_->val.y < val[index].y) {
      root_->right = new Vertex(val[index], index + 1);
      root_->right->parent = root_;
      root_ = root_->right;
    } else {
      Vertex* cur = root_;
      while (cur->parent != nullptr && cur->val.y >= val[index].y) {
        cur = cur->parent;
      }
      if (cur->val.y >= val[index].y) {
        root_ = new Vertex(val[index], index + 1);
        root_->left = cur;
        cur->parent = root_;
      } else {
        root_ = new Vertex(val[index], index + 1);
        root_->parent = cur;
        root_->left = cur->right;
        if (cur->right != nullptr) {
          cur->right->parent = root_;
        }
        cur->right = root_;
      }
    }
  }

  while (root_->parent != nullptr) {
    root_ = root_->parent;
  }
}

void Treap::PrintVertex() const {
  std::vector<std::pair<size_t, std::pair<size_t, size_t>>> ver(size_);
  if (root_ != nullptr) {
    PrintRec(root_, ver);
  }
  for (const auto& p : ver) {
    std::cout << p.first << ' ' << p.second.first << ' ' << p.second.second
              << '\n';
  }
}

void Treap::PrintRec(
    Vertex* current,
    std::vector<std::pair<size_t, std::pair<size_t, size_t>>>& ver) const {
  if (current->parent != nullptr) {
    ver[current->num - 1].first = current->parent->num;
  }
  if (current->left != nullptr) {
    ver[current->num - 1].second.first = current->left->num;
    PrintRec(current->left, ver);
  }
  if (current->right != nullptr) {
    ver[current->num - 1].second.second = current->right->num;
    PrintRec(current->right, ver);
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.precision(20);

  size_t n;
  std::cin >> n;

  std::vector<Point> priority(n);
  std::cin >> priority;

  if (std::unique(priority.begin(), priority.end()) != priority.end()) {
    std::cout << "NO\n";
  } else {
    std::cout << "YES\n";
    Treap treap(priority);
    treap.PrintVertex();
  }
  
  return 0;
}
