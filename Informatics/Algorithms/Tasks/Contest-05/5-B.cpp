/*

B (2 балла, с ревью). Заниженная граница (AVL)

Ограничение времени	3 секунды
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Доминик проводит инвентаризацию (опять). Он уверен, что машины можно и нужно
сравнить по числу миллисекунд для разгона от нуля до 100 км/ч. Он просит вас
помочь ему, а именно, ваш алгоритм должен работать с множеством машин Доминика.
Нужно реализовать следующие операции:

add(i)  — добавить в множество машин тачку с разгоном в i миллисекунд (если
такая там уже есть, то Доминик отвлекся на семейные разговоры и подсунул вам
второй раз ту же машину);
next(i)  — узнать машину с минимальным временем разгона, не меньшим i. Если
искомая машина отсутствует, необходимо вывести -1 и попросить Доминика быть
внимательнее.

Формат ввода
Исходно множество машин пусто. Первая строка входного файла содержит число n —
количество запросов Доминика (1 < n < 3 ⋅ 105). Следующие n строк содержат
операции. Каждая операция имеет вид:
+ i  — add(i)
? i  — next(i)
Если операция + идет во входном файле в начале или после другой операции +, то
она задает операцию add(i). Если же она идет после запроса ?, и результат этого
запроса был y, то выполняется операция add((i + y) % 1e9). Это нужно, чтобы
Доминик убедился в том, что вы достойный член семьи и не реализовали оффлайн
алгоритм.
Во всех запросах и операциях добавления параметры лежат в интервале от 0 до 1e9.

Формат вывода
Для каждого запроса выведите одно число — ответ на запрос.

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
  Type Get(Type value) const;

 private:
  struct Vertex;

  Vertex* root_;
  size_t size_;

  Vertex* InsertInternal(Vertex* vertex, Type value);
  Type GetInternal(const Vertex* vertex, Type value) const;

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
  mutable size_t height;

  int GetBalanceFactor() const;
  void UpdateHeight() const;
};

template <typename Type>
AVLTree<Type>::Vertex::Vertex(Type value_new)
    : value(value_new), left(nullptr), right(nullptr), height(1) {}

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
AVLTree<Type>::~AVLTree() {
  delete root_;
}

template <typename Type>
struct AVLTree<Type>::Vertex* AVLTree<Type>::RightRotate(Vertex* vertex) {
  Vertex* left = vertex->left;
  vertex->left = left->right;
  left->right = vertex;
  vertex->UpdateHeight();
  left->UpdateHeight();
  return left;
}

template <typename Type>
struct AVLTree<Type>::Vertex* AVLTree<Type>::LeftRotate(Vertex* vertex) {
  Vertex* right = vertex->right;
  vertex->right = right->left;
  right->left = vertex;
  vertex->UpdateHeight();
  right->UpdateHeight();
  return right;
}

template <typename Type>
struct AVLTree<Type>::Vertex* AVLTree<Type>::BalanceTree(Vertex* vertex) {
  vertex->UpdateHeight();
  if (vertex->GetBalanceFactor() > 1) {
    if (vertex->right->GetBalanceFactor() < 0) {
      vertex->right = RightRotate(vertex->right);
    }
    return LeftRotate(vertex);
  }
  if (vertex->GetBalanceFactor() < -1) {
    if (vertex->left->GetBalanceFactor() < 0) {
      vertex->left = RightRotate(vertex->left);
    }
    return RightRotate(vertex);
  }
  return vertex;
}

template <typename Type>
void AVLTree<Type>::Insert(Type value) {
  root_ = InsertInternal(root_, value);
}

template <typename Type>
Type AVLTree<Type>::Get(Type value) const {
  return GetInternal(root_, value);
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
Type AVLTree<Type>::GetInternal(const Vertex* vertex, Type value) const {
  if (vertex == nullptr) {
    return -1;
  }

  if (vertex->value < value) {
    return GetInternal(vertex->right, value);
  }

  Type left = GetInternal(vertex->left, value);
  if (left == -1) {
    return vertex->value;
  }
  return (vertex->value < left) ? vertex->value : left;
}

void ProcessQueries(size_t n) {
  static const size_t kMod = 1'000'000'000;
  AVLTree<long long> tree;

  long long last_operation = -2;
  for (size_t index = 0; index < n; ++index) {
    char type;
    long long key;
    std::cin >> type >> key;

    if (type == '+') {
      key = (last_operation == -2) ? key : (key + last_operation + kMod) % kMod;
      if (tree.Get(key) != key) {
        tree.Insert(key);
      }
      last_operation = -2;
    } else if (type == '?') {
      last_operation = tree.Get(key);
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
