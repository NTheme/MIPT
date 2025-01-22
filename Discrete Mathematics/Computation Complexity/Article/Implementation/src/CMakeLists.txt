# --========================================-- #
#   * Author  : NTheme - All rights reserved
#   * Created : 1 December 2024, 4:18 AM
#   * File    : CMakeLists.txt
#   * Project : Salesman
# --========================================-- #

add_subdirectory(BlossomV)

add_library(tsp_approximation STATIC TSP.cpp)
target_include_directories(tsp_approximation PUBLIC "${HEADER_DIR}")
target_link_libraries(tsp_approximation PUBLIC blossom_v)
