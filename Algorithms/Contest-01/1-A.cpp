/*
A (2 балла). Стек

Ограничение времени	1 секунда
Ограничение памяти	256Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt

Реализуйте свой стек.

Решения, использующие std::stack, получат 1 балл. Решения, хранящие стек в
массиве, получат 1.5 балла. Решения, использующие указатели, получат 2 балла.

Гарантируется, что количество элементов в стеке ни в какой момент времени не
превышает 10000.

Обработайте следующие запросы:
push n: добавить число n в конец стека и вывести «ok»;
pop: удалить из стека последний элемент и вывести его значение, либо вывести
«error», если стек был пуст; back: сообщить значение последнего элемента стека,
либо вывести «error», если стек пуст; size: вывести количество элементов в
стеке; clear: опустошить стек и вывести «ok»; exit: вывести «bye» и завершить
работу.

Формат ввода
В каждой строке входных данных задана одна операция над стеком в формате,
описанном выше.

Формат вывода
В ответ на каждую операцию выведите соответствующее сообщение.
*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

#include <iostream>

namespace NT {
template <typename Type>
struct Stack {
 public:
  Stack() : root_(NULL), size_(0) {}

  ~Stack() {
    while (root_ != NULL) {
      Elem* del = root_;
      root_ = root_->next;
      delete del;
    }
    std::cout << "bye\n";
  }

  size_t Size() { return size_; }

  void Push(Type n) {
    root_ = new Elem(root_, n);
    ++size_;
    std::cout << "ok\n";
  }

  void Pop() {
    if (root_ == NULL) {
      std::cout << "error\n";
    } else {
      Elem* del = root_;
      root_ = root_->next;
      std::cout << del->data << "\n";
      delete del;
      --size_;
    }
  }

  void Back() {
    if (root_ == NULL) {
      std::cout << "error\n";
    } else {
      std::cout << root_->data << '\n';
    }
  }

  void Clear() {
    while (root_ != NULL) {
      Elem* del = root_;
      root_ = root_->next;
      delete del;
      --size_;
    }
    std::cout << "ok\n";
  }

 private:
  struct Elem {
    Elem* next;
    Type data;

    Elem() : next(nullptr), data(NULL) {}
    Elem(Elem* next, const Type& val) : next(next), data(val) {}
  };

  Elem* root_;
  size_t size_;
};
}  // namespace NT

void StackQuery() {
  NT::Stack<long long> stack;
  while (true) {
    std::string s;
    std::cin >> s;

    if (s == "push") {
      int buf = 0;
      std::cin >> buf;
      stack.Push(buf);
    } else if (s == "pop") {
      stack.Pop();
    } else if (s == "back") {
      stack.Back();
    } else if (s == "size") {
      std::cout << stack.Size() << '\n';
    } else if (s == "clear") {
      stack.Clear();
    } else if (s == "exit") {
      break;
    }
  }
}

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(0), std::cout.tie(0);

  StackQuery();

  return 0;
}
