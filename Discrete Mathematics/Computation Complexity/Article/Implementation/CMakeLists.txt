# --========================================-- #
#   * Author  : NTheme - All rights reserved
#   * Created : 1 December 2024, 4:18 AM
#   * File    : CMakeLists.txt
#   * Project : Salesman
# --========================================-- #

cmake_minimum_required(VERSION 3.24)

project(ComplexityProject)
set(HEADER_DIR "${PROJECT_SOURCE_DIR}/include")

add_subdirectory(src)

add_executable(main main.cpp)
target_link_libraries(main PUBLIC tsp_approximation)
