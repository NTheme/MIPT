build/task: build/main.o build/date.o
	gcc -fsanitize=address build/main.o build/date.o -o build/task

build/date.o: date.c date.h
	gcc -Wall -Ofast -unroll-loops -fno-stack-protector -fsanitize=address -c date.c -o build/date.o

build/main.o: main.c date.h
	gcc -Wall -Ofast -unroll-loops -fno-stack-protector -fsanitize=address -c main.c -o build/main.o