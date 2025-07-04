# MIPT formal task: Regular expressions

## Build & Run
```shell
cmake . -B build
cmake --build build
./build/proj
```

## Test coverage
```shell
cmake . -B build -DCMAKE_BUILD_TYPE=Debug -DTEST=0
cd build
make test_coverage
```
Coverage report will appear in ```build/coverage``` (of course if you're not me and lucky enough to deal with lcov...)

## Proof
У нас дано регулярное выражение в обратной польской записи. Разбираем его и превращаем в граф.
1) Если символ - буква, то создаем 2 новые вершины (начало и конец) и переход между ними в виде этой буквы. Кладем на стек
2) Если символ - плюс, то соединяем последний подграф с предпоследним, параллельно (начало 1ого -> начало 2ого по пустому переходу, конец 2ого -> конец 1ого по пустому переходу)
3) Если символ - звездочка, то зацикливаем на себя: начало с концом, конец с началом у последней пары начало-конец на стеке
4) Если символ - точка, то соединяем конец предпоследнего с началом последнего и последний убираем со стека

Корректность разбора тривиальна. Мы получили автомат. Теперь пройдемся по нему, чтобы понять, каких остатков по модулю k мы можем добиться при входе в терминальную вершину.
Для этого запустимся обычным BFS, но считая, что у нас вершин каждого типа не 1, а k (в зависимости от того, с каким остатком в нее пришли). База - корень_дерева-дистация_0.
Ответ будет лежать в остатке l. Если нашелся, то победили, иначе -1, то есть, невозможно. ЧТД

---
By ***NTheme***