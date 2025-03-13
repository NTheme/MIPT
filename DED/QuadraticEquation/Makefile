DED_FLAGS = `cat flags.txt`

build/task: build/main.o build/quadratic.o build/test.o build/common.o
	g++ $(DED_FLAGS) build/main.o build/quadratic.o build/test.o build/common.o -o build/task

build/main.o: main.cpp quadratic.h test.h common.h
	g++ $(DED_FLAGS) -c main.cpp -o build/main.o

build/quadratic.o: quadratic.cpp quadratic.h common.h
	g++ $(DED_FLAGS) -c quadratic.cpp -o build/quadratic.o

build/test.o: test.cpp quadratic.h common.h
	g++ $(DED_FLAGS) -c test.cpp -o build/test.o

build/common.o: common.cpp common.h
	g++ $(DED_FLAGS) -c common.cpp -o build/common.o