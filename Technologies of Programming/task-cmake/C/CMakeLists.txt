cmake_minimum_required(VERSION 3.0)
project("gtest")

set(SOURCES main_test.cpp)
set(HEADERS tests.h)

enable_testing()
add_executable(CTest main_test.cpp)
target_link_libraries(CTest gtest megagigachad_library)

