/*
NOT WORKING!!!!!!!!!!!!!!!!!!!!!!!!!

K (4 балла, с ревью). Биномиальная куча

Ограничение времени	2 секунды
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Реализуйте биномиальную кучу.

Формат ввода
В первой строке содержится два целых числа: N — общее количество куч и M —
количество операций (1 ≤ N ≤ 1000, 1 ≤ M ≤ 1000000). Изначально все кучи пусты.
Требуется поддерживать следующие операции:
0 a v — добавить элемент со значением v в кучу с номером a. Вновь добавленный
элемент имеет уникальный индекс равный порядковому номеру соответствующей
операции добавления. Нумерация начинается с единицы.
1 a b — переложить все элементы из кучи с номером a в кучу с номером b. После
этой операции куча a становится пустой. 2 i — удалить элемент с индексом i. 3 i
v — присвоить элементу с индексом i значение v. Гарантируется, что элемент
существует. 4 a — вывести на отдельной строке значение минимального элемента в
куче с номером a. Гарантируется, что куча не пуста. 5 a — удалить минимальный
элемент из кучи с номером a. Если таковых несколько, то выбирается элемент с
минимальным индексом. Гарантируется, что куча не пуста.

Формат вывода
Для каждой операции поиска
минимального элемента выведите единственное число: значение искомого элемента.
*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <iostream>
#include <vector>

namespace NT {
template <typename Type>
class Vertex {
 public:
  Vertex(Type val_new, size_t index_new)
      : val(val_new), index(index_new), parent(nullptr) {}
  ~Vertex() {
    for (auto p : childs) {
      delete p;
    }
  }

  Type val;
  size_t index;
  Vertex *parent;
  std::vector<Vertex *> childs;

  size_t Rank() const { return childs.size(); }
};

template <typename Type>
class BinomialHeap {
 public:
  BinomialHeap() : size_(0) {}
  BinomialHeap(Vertex<Type> *added) : size_(1) { trees_.push_back(added); }
  BinomialHeap(std::vector<Vertex<Type> *> trees_new) : trees_(trees_new) {}
  ~BinomialHeap() {
    for (auto p : trees_) {
      delete p;
    }
  }

  void MergeHeaps(BinomialHeap &added_heap);
  Vertex<Type> *Insert(Type val, size_t index);
  Type GetMin() const;
  void EraseMin();
  size_t Size() const;
  void Clear();

 private:
  size_t size_;
  std::vector<Vertex<Type> *> trees_;
  std::vector<Vertex<Type> *> ptr_;

  void AppendTree(std::vector<Vertex<Type> *> &res_trees,
                  Vertex<Type> *added_tree);
  Vertex<Type> *MergeTrees(Vertex<Type> *res_tree, Vertex<Type> *added_tree);
};

template <typename Type>
Vertex<Type> *BinomialHeap<Type>::MergeTrees(Vertex<Type> *res_tree,
                                             Vertex<Type> *added_tree) {
  if (res_tree->val > added_tree->val) {
    std::swap(res_tree, added_tree);
  }
  res_tree->childs.push_back(added_tree);
  added_tree->parent = res_tree;
  return res_tree;
}

template <typename Type>
void BinomialHeap<Type>::AppendTree(std::vector<Vertex<Type> *> &res_trees,
                                    Vertex<Type> *added_tree) {
  if (!res_trees.empty() && res_trees.back()->Rank() == added_tree->Rank()) {
    res_trees.back() = MergeTrees(res_trees.back(), added_tree);
  } else {
    res_trees.push_back(added_tree);
  }
}

template <typename Type>
void BinomialHeap<Type>::MergeHeaps(BinomialHeap<Type> &added_heap) {
  std::vector<Vertex<Type> *> res_trees;

  size_t i = 0;
  size_t j = 0;
  while (i < trees_.size() && j < added_heap.trees_.size()) {
    Vertex<Type> *buffer = nullptr;
    if (trees_[i]->Rank() == added_heap.trees_[j]->Rank()) {
      buffer = MergeTrees(trees_[i++], added_heap.trees_[j++]);
    } else if (trees_[i]->Rank() < added_heap.trees_[j]->Rank()) {
      buffer = trees_[i++];
    } else {
      buffer = added_heap.trees_[j++];
    }
    AppendTree(res_trees, buffer);
  }

  while (i < trees_.size()) {
    AppendTree(res_trees, trees_[i++]);
  }

  while (j < added_heap.trees_.size()) {
    AppendTree(res_trees, added_heap.trees_[j++]);
  }

  size_ += added_heap.Size();
  trees_ = res_trees;
  added_heap.Clear();
}

template <typename Type>
Vertex<Type> *BinomialHeap<Type>::Insert(Type val, size_t index) {
  Vertex<Type> *added = new Vertex<Type>(val, index);
  BinomialHeap<Type> new_heap(added);
  MergeHeaps(new_heap);
  return added;
}

template <typename Type>
Type BinomialHeap<Type>::GetMin() const {
  Type min_val = trees_[0]->val;
  for (size_t index = 1; index < trees_.size(); ++index) {
    min_val = std::min(min_val, trees_[index]->val);
  }

  return min_val;
}

template <typename Type>
void BinomialHeap<Type>::EraseMin() {
  size_t erase_ind = 0;
  for (size_t index = 1; index < trees_.size(); ++index) {
    erase_ind =
        (trees_[erase_ind]->val < trees_[index]->val) ? erase_ind : index;
  }

  for (auto &p : trees_[erase_ind]->childs) {
    p->parent = nullptr;
  }
  BinomialHeap new_heap(trees_[erase_ind]->childs);
  trees_.erase(trees_.begin() + erase_ind);
  MergeHeaps(new_heap);
}

template <typename Type>
void BinomialHeap<Type>::Clear() {
  trees_.clear();
}

template <typename Type>
size_t BinomialHeap<Type>::Size() const {
  return size_;
}
}  // namespace NT

void AddElement(std::vector<NT::BinomialHeap<long long>> &heaps,
                std::vector<std::pair<size_t, NT::Vertex<long long> *>> &values,
                size_t ind) {
  size_t a = 0;
  long long v = 0;
  std::cin >> a >> v;
  values.emplace_back(a, heaps[a - 1].Insert(v, ind));
}

void MoveElements(std::vector<NT::BinomialHeap<long long>> &heaps) {
  size_t a = 0, b = 0;
  std::cin >> a >> b;
  heaps[b - 1].MergeHeaps(heaps[a - 1]);
}

void Dld(std::vector<NT::BinomialHeap<long long>> &heaps,
         std::vector<std::pair<size_t, NT::Vertex<long long> *>> &values,
         size_t ind) {
  NT::Vertex<long long> *erase = values[ind - 1].second;

  while (erase->parent != nullptr) {
    std::swap(values[erase->index - 1], values[erase->parent->index - 1]);
    std::swap(erase, erase->parent);
  }

  erase->val = -1e18;
  heaps[values[ind - 1].first].EraseMin();
}

void DeleteElement(
    std::vector<NT::BinomialHeap<long long>> &heaps,
    std::vector<std::pair<size_t, NT::Vertex<long long> *>> &values) {
  size_t ind = 0;
  std::cin >> ind;
  Dld(heaps, values, ind);
}

void ChangeElement(
    std::vector<NT::BinomialHeap<long long>> &heaps,
    std::vector<std::pair<size_t, NT::Vertex<long long> *>> &values,
    size_t index) {
  size_t ind = 0;
  long long val = 0;
  std::cin >> ind >> val;
  Dld(heaps, values, ind);
  heaps[values[ind - 1].first - 1].Insert(val, index);
}

void PrintMinElement(std::vector<NT::BinomialHeap<long long>> &heaps) {
  size_t a = 0;
  std::cin >> a;
  std::cout << heaps[a - 1].GetMin() << '\n';
}

void DeleteMinElement(std::vector<NT::BinomialHeap<long long>> &heaps) {
  size_t a = 0;
  std::cin >> a;
  heaps[a - 1].EraseMin();
}

void ProcessQueries(
    std::vector<NT::BinomialHeap<long long>> &heaps,
    std::vector<std::pair<size_t, NT::Vertex<long long> *>> &values,
    size_t ind) {
  size_t type = -1;
  std::cin >> type;

  switch (type) {
    case 0:
      AddElement(heaps, values, ind);
      break;
    case 1:
      MoveElements(heaps);
      break;
    case 2:
      DeleteElement(heaps, values);
      break;
    case 3:
      ChangeElement(heaps, values, ind);
      break;
    case 4:
      PrintMinElement(heaps);
      break;
    case 5:
      DeleteMinElement(heaps);
      break;
    default:
      break;
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);

  size_t n = 0, m = 0;
  std::cin >> n >> m;

  std::vector<NT::BinomialHeap<long long>> heaps(n);
  std::vector<std::pair<size_t, NT::Vertex<long long> *>> values;

  for (size_t ind = 0; ind < n; ++ind) {
    ProcessQueries(heaps, values, ind);
  }

  return 0;
}
