/*

F (2 балла). Подсемья и ее сумма

Ограничение времени	2 секунды
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

«Нельзя отворачиваться от семьи. Даже если она отвернулась от тебя» (c)
Форсаж 6. Все дороги принадлежат им!

Вы решили собрать всю семью вместе, чтобы узнать, как у них настрой. Пока еще
никто не приехал (семья же отвернулась), но вот-вот они будут тут. Вы хотите
понимать, каково суммарное настроение у семьи и ее подсемей. Для этого вы хотите
уметь следующие вещи:

add(i) — обработать приезд нового члена семьи с настроем i (если он там уже
есть, то состав приехавших не изменился, и было введено ошибочное значение);
sum(l, r) — вывести суммарное настроение всех членов с настроем x, которые
удовлетворяют неравенству l ≤ x ≤ r.

Формат ввода
Изначально никто не прибыл еще. Первая строка входного файла содержит n —
количество запросов (1 ≤ n ≤ 300 000). Следующие n строк содержат операции.
Каждая операция имеет вид либо «+ i», либо «? l r». Операция «? l r» задает
запрос sum(l, r).

Если операция «+ i» идет во входном файле в начале или после другой операции
«+», то она задает операцию add(i). Если же она идет после запроса «?», и
результат этого запроса был y, то выполняется операция add((i + y) % 1e9).

Во всех запросах и операциях добавления параметры лежат в интервале от 0 до
10^9.

*/
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <vector>

template <typename Type>
class AVLTree {
 public:
  AVLTree() : root_(nullptr), size_(0) {}
  ~AVLTree();

  void Insert(Type value);
  Type Get(Type right) const;

 private:
  struct Vertex;

  Vertex* root_;
  size_t size_;

  Vertex* InsertInternal(Vertex* vertex, Type value);
  Type GetInternal(const Vertex* vertex, Type right) const;

  Vertex* LeftRotate(Vertex* vertex);
  Vertex* RightRotate(Vertex* vertex);
  Vertex* BalanceTree(Vertex* vertex);
};

template <typename Type>
struct AVLTree<Type>::Vertex {
  Vertex(Type value_new);
  ~Vertex();

  Type value;
  Vertex* left;
  Vertex* right;
  mutable Type sum;
  mutable size_t height;

  int GetBalanceFactor() const;
  void UpdateHeight() const;
  void UpdateSum() const;
};

template <typename Type>
AVLTree<Type>::Vertex::Vertex(Type value_new)
    : value(value_new),
      left(nullptr),
      right(nullptr),
      sum(value_new),
      height(1) {}

template <typename Type>
AVLTree<Type>::Vertex::~Vertex() {
  delete left;
  delete right;
}

template <typename Type>
int AVLTree<Type>::Vertex::GetBalanceFactor() const {
  size_t height_left = (left == nullptr) ? 0 : left->height;
  size_t height_right = (right == nullptr) ? 0 : right->height;
  return height_right - height_left;
}

template <typename Type>
void AVLTree<Type>::Vertex::UpdateHeight() const {
  size_t height_left = (left == nullptr) ? 0 : left->height;
  size_t height_right = (right == nullptr) ? 0 : right->height;
  height = std::max(height_left, height_right) + 1;
}

template <typename Type>
void AVLTree<Type>::Vertex::UpdateSum() const {
  Type sum_left = (left == nullptr) ? 0 : left->sum;
  Type sum_right = (right == nullptr) ? 0 : right->sum;
  sum = sum_left + sum_right + value;
}

template <typename Type>
AVLTree<Type>::~AVLTree() {
  delete root_;
}

template <typename Type>
struct AVLTree<Type>::Vertex* AVLTree<Type>::RightRotate(Vertex* vertex) {
  Vertex* left = vertex->left;
  vertex->left = left->right;
  left->right = vertex;
  vertex->UpdateSum();
  vertex->UpdateHeight();
  left->UpdateSum();
  left->UpdateHeight();
  return left;
}

template <typename Type>
struct AVLTree<Type>::Vertex* AVLTree<Type>::LeftRotate(Vertex* vertex) {
  Vertex* right = vertex->right;
  vertex->right = right->left;
  right->left = vertex;
  vertex->UpdateSum();
  vertex->UpdateHeight();
  right->UpdateSum();
  right->UpdateHeight();
  return right;
}

template <typename Type>
struct AVLTree<Type>::Vertex* AVLTree<Type>::BalanceTree(Vertex* vertex) {
  vertex->UpdateHeight();
  if (vertex->GetBalanceFactor() == 2) {
    if (vertex->right->GetBalanceFactor() < 0) {
      vertex->right = RightRotate(vertex->right);
    }
    return LeftRotate(vertex);
  }
  if (vertex->GetBalanceFactor() == -2) {
    if (vertex->left->GetBalanceFactor() < 0) {
      vertex->left = RightRotate(vertex->left);
    }
    return RightRotate(vertex);
  }
  vertex->UpdateSum();
  return vertex;
}

template <typename Type>
void AVLTree<Type>::Insert(Type value) {
  root_ = InsertInternal(root_, value);
}

template <typename Type>
Type AVLTree<Type>::Get(Type right) const {
  return GetInternal(root_, right);
}

template <typename Type>
struct AVLTree<Type>::Vertex* AVLTree<Type>::InsertInternal(Vertex* vertex,
                                                            Type value) {
  if (vertex == nullptr) {
    return new Vertex(value);
  }
  if (vertex->value < value) {
    vertex->right = InsertInternal(vertex->right, value);
  } else {
    vertex->left = InsertInternal(vertex->left, value);
  }
  return BalanceTree(vertex);
}

template <typename Type>
Type AVLTree<Type>::GetInternal(const Vertex* vertex, Type right) const {
  if (vertex == nullptr) {
    return 0;
  }

  if (vertex->value >= right) {
    return GetInternal(vertex->left, right);
  }
  return vertex->value + (vertex->left ? vertex->left->sum : 0) +
         GetInternal(vertex->right, right);
}

void ProcessQueries(size_t n) {
  static const size_t kMod = 1'000'000'000;
  AVLTree<long long> tree;

  long long last_operation = -2;
  for (size_t index = 0; index < n; ++index) {
    char type;
    std::cin >> type;

    if (type == '+') {
      long long value;
      std::cin >> value;
      value = (last_operation == -2) ? value
                                     : (value + last_operation + kMod) % kMod;
      if (tree.Get(value + 1) - tree.Get(value) != value) {
        tree.Insert(value);
      }
      last_operation = -2;
    } else if (type == '?') {
      size_t left, right;
      std::cin >> left >> right;
      last_operation = tree.Get(right + 1) - tree.Get(left);
      std::cout << last_operation << '\n';
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
