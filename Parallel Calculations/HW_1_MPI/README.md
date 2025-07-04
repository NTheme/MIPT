# Задача №1. Нахождение интеграла с использованием MPI

Исправлять коды после комментариев можно в течение месяца после проверки.

## $`\int\limits_0^1\frac{4}{1+x^2}dx`$

### Постановка задачи: решить определенный интеграл методом трапеций

Предполагается, что запуск исполняемого файла будет происходить с использованием p процессов.
Один из p процессов («основной») разбивает отрезок $`[0; 1]`$ на $`N`$ малых отрезков длиной $`\Delta x`$ (шаг интегрирования),
и вычисляет с этим разбиением интеграл в последовательном варианте. Далее этот же процесс разбивает отрезок $`[0; 1]`$,
состоящий из $`N`$ малых отрезков, на $`p`$ частей и границы каждой из оставшихся $`(p-1)`$ частей рассылает остальным $`(p-1)`$ процессам
(с одной из частей отрезка работает сам «основной» процесс). Число $`N`$ может меняться и задается пользователем.

Каждый из процессов, получивших свои границы части отрезка, должен вычислить свою часть интеграла $`I_i`$ и отправить ее «основному» процессу.

«Основной» процесс получает все части интеграла от процессов-рабочих и, складывая их, получает исходный интеграл  $`I`$.

### Задание

1. Вывести на экран в столбик значения частей интеграла $`I_i`$, посчитанные каждым из процессов-рабочих с указанием его номера.
2. Вывести на экран значение интеграла $`I`$, посчитанное сложением всех частей интеграла, полученных «основным» процессом от процессов-рабочих.
3. Вывести на экран интеграл $`I_0`$, посчитанный «основным» процессом последовательно. Сравнить его со значением  $`I`$.
4. На одной координатной плоскости построить 3 графика зависимости ускорения $`S`$ от количества процессов $`p`$, где $`p = 1,2,3, \ldots ,8`$ для $`N = 1000, N = 10^6`$  и для $`N = 10^8`$.
5. На основе графиков написать вывод о полученных результатах и попытаться обосновать их

### Вывод

Вывод: с ростом $N$ повышается точность, которую мы хотим получить от подсчета интеграла в силу метода Монте-Карло. Соответственно, с ростом требуемой точности повышается степень, до которой можно распараллелить программу. Это соответствует ожиданиям, так как отношение времени параллельной работы к последовательной увеличивается. Для точности $10^3$ мы наблюдаем отсутствие пользы в парллельном вычислении, даже некоторый вред, так как количество частей слишком мало и больше времени уходит на создание процессов и работу с памятью. Для других значений $N$ параллельное вычисление имеет смысл, мы видим примерно линейную зависимость ускорения от количества процессов. Можно определить оптимальное количество процессов для каждой точности: 1 для $10^3$ частей и 8 для $10^6$ и $10^8$.

### Примечания

1. Подумать над разбивкой отрезка $`[0; 1]`$  на части, когда $`N`$ не делится нацело на $`p`$.
2. Использовать следующую особенность программирования: если не обращаться ни к какому из процессов посредством конструкции if, а написать код в общей части, то этот кусок кода будет выполнен всеми процессами одинаково.
3. Если в общей части программы объявить переменную, то она в разных процессах будет называться одинаково, но может при этом принимать различные значения.

### Как сдать задание

1. Каждому из вас на почту пришел доступ на http://gitlab.atp-fivt.org/. (если всё ещё не пришел - стучитесь [сюда](https://forms.gle/1jJuD3StgKuy8MVKA)). В http://gitlab.atp-fivt.org/ создан репозиторий `<ваше_ФИО>-mpi`.
2. Для каждой задачи (в данном случае она всего 1) для вас создана ветка, в которую нужно заливать решение.
3. Для сдачи задания нужно до deadline сделать pull request в master. В нем должен быть
    - код решения
    - графики (в виде img, png или pdf)
    - код, позволяющий воспроизвести эксперимент для получения графика.
4. После deadline проверяющий (это может быть случайный семинарист или ассистент) оставляет комментарии и ставим *текущую оценку*. Если решение вцелом работает правильно и соответсвует требованиям, его можно исправлять (см. deadline по исправлениям) и повысить таким образом оценку.
