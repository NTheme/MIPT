cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_module)

project(FormalTask)

if (DEFINED TEST)
  set(CMAKE_CXX_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage --coverage")
  find_package(GTest REQUIRED)
  include(CodeCoverage)
  add_executable(proj test.cpp)
  target_link_libraries(proj GTest::gtest GTest::gtest_main)
  setup_target_for_coverage(test_coverage proj coverage)
else()
  add_executable(proj AutomatonTask.cpp)
endif ()
