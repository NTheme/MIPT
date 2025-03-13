/*
E (3 балла, с ревью). Распределяющая шляпа
Ограничение времени	2 секунды
Ограничение памяти	64Mb
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt
Надел Поттер распределяющую шляпу, а ему Слизерин как раз. (с) Мои любимые
потеррески.

Распределяющая шляпа — крайне древний артефакт, способный по ее носителю понять,
какой факультет ему подойдет для наиболее полного раскрытия характера. Но некто
решил заколдовать шляпу, теперь она определяет уровень IQ носителя. Вам
предстоит реализовать заколдованную шляпу, чтобы с оригинальной сняли проклятье.

У вас есть сама шляпа и набор действий, который будет с ней происходить. К шляпе
стоит очередь из первокурсников Хогвартса, которая желает протестировать себя.
Возможные действия:

«enqueue n» Добавить в внутреннюю очередь шляпы уровень интеллекта очередного
первокурсника n (1 ≤ n ≤ 10^9) (значение n задается после команды). Шляпа должна
сказать «ok». «dequeue» Удалить из внутренней очереди шляпы уровень интеллекта
последнего студента, которого она еще помнит. Шляпа должна сказать его значение.
«front» Шляпа должна сказать уровень интеллекта последнего студента, которого
она еще помнит, не забывая его. «size» Шляпа скажет, уровень интеллекта какого
числа студентов она помнит. «clear» Перезагрузка шляпы, она забывает все, что
было до этого. Шляпа должна сказать «ok». «min» Шляпа должна сказать уровень
интеллекта самого неодаренного умственными способностями первокурсника. При
этом, конечно же, не забыть его. Перед исполнением операций «front», «dequeue» и
«min» шляпа должна проверять, содержится ли в внутренней очереди хотя бы один
элемент. Если шляпа помнит ноль студентов на момент таких запросов, то она
должна вместо числового значения сказать слово «error».

Формат ввода
В первой строке входных данных записано единственное число M (1 ≤ M ≤ 2 ⋅ 105) —
количество команд. В следующих М строках дано по одной команде из тех, что идут
выше.

Формат вывода
Для каждой команды выведите одну строчку — то, что скажет шляпа.
*/

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("no-stack-protector")

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

namespace NT {
template <typename Type>
class Stack {
 public:
  Stack() : root_(nullptr), size_(0) {}
  ~Stack() { Clear(); }

  size_t Size();
  void Push(Type n);
  Type Back();
  Type Pop();
  void Clear();

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

template <typename Type>
void Stack<Type>::Push(Type n) {
  root_ = new Elem(root_, n);
  ++size_;
}

template <typename Type>
size_t Stack<Type>::Size() {
  return size_;
}

template <typename Type>
Type Stack<Type>::Back() {
  return root_->data;
}

template <typename Type>
Type Stack<Type>::Pop() {
  Elem* del = root_;
  root_ = root_->next;
  --size_;

  Type removed = del->data;

  delete del;
  return removed;
}

template <typename Type>
void Stack<Type>::Clear() {
  while (root_ != nullptr) {
    Elem* del = root_;
    root_ = root_->next;
    delete del;
    --size_;
  }
}

template <typename Type>
class Queue {
 public:
  Queue() {}
  ~Queue() {}

  size_t Size();
  void Push(Type n);
  Type Pop();
  Type Front();
  void Clear();
  Type Min();

 private:
  Stack<std::pair<Type, Type>> left_, right_;
  void Balance();
};

template <typename Type>
size_t Queue<Type>::Size() {
  return left_.Size() + right_.Size();
}

template <typename Type>
void Queue<Type>::Push(Type n) {
  Type minimal = (left_.Size() == 0) ? n : std::min(n, left_.Back().second);
  left_.Push({n, minimal});
}

template <typename Type>
Type Queue<Type>::Pop() {
  Balance();

  if (right_.Size() == 0) {
    return 0;
  }
  return right_.Pop().first;
}

template <typename Type>
Type Queue<Type>::Front() {
  Balance();

  if (right_.Size() == 0) {
    return 0;
  }
  return right_.Back().first;
}

template <typename Type>
void Queue<Type>::Clear() {
  while (left_.Size() > 0) {
    left_.Pop();
  }
  while (right_.Size() > 0) {
    right_.Pop();
  }
}

template <typename Type>
Type Queue<Type>::Min() {
  Balance();

  if (Size() == 0) {
    return 0;
  }
  if (left_.Size() == 0) {
    return right_.Back().second;
  }
  return std::min(left_.Back().second, right_.Back().second);
}

template <typename Type>
void Queue<Type>::Balance() {
  if (right_.Size() == 0) {
    while (left_.Size() > 0) {
      Type elem = left_.Pop().first;
      int minimal =
          (right_.Size() == 0) ? elem : std::min(elem, right_.Back().second);
      right_.Push({elem, minimal});
    }
  }
}
}  // namespace NT

void PrintValue(int val) {
  if (val == 0) {
    std::cout << "error\n";
  } else {
    std::cout << val << '\n';
  }
}

void QueueQuery(NT::Queue<int>& queue) {
  std::string s;
  std::cin >> s;

  if (s == "enqueue") {
    int buf = 0;
    std::cin >> buf;
    queue.Push(buf);
    std::cout << "ok\n";
  } else if (s == "dequeue") {
    PrintValue(queue.Pop());
  } else if (s == "front") {
    PrintValue(queue.Front());
  } else if (s == "size") {
    std::cout << queue.Size() << '\n';
  } else if (s == "clear") {
    queue.Clear();
    std::cout << "ok\n";
  } else if (s == "min") {
    PrintValue(queue.Min());
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(0), std::cout.tie(0);
  std::cout.precision(20);
  std::srand((unsigned int)time(NULL));

  size_t n = 0;
  std::cin >> n;

  NT::Queue<int> queue;
  while (n-- > 0) {
    QueueQuery(queue);
  }

  return 0;
}
