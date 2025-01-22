# --========================================-- #
#   * Author  : NTheme - All rights reserved
#   * Created : 1 December 2024, 4:18 AM
#   * File    : CMakeLists.txt
#   * Project : Salesman
# --========================================-- #

set(BLOSSOM_HEADERS "${HEADER_DIR}/BlossomV")

add_library(mincost STATIC
        MinCost/MinCost.cpp
)
target_include_directories(mincost PUBLIC "${BLOSSOM_HEADERS}/MinCost")

add_library(perfect_matching STATIC
        misc.cpp
        PMduals.cpp
        PMexpand.cpp
        PMinit.cpp
        PMinterface.cpp
        PMmain.cpp
        PMrepair.cpp
        PMshrink.cpp
)
target_include_directories(perfect_matching PUBLIC "${BLOSSOM_HEADERS}")

add_library(blossom_v INTERFACE)
target_link_libraries(blossom_v INTERFACE perfect_matching mincost)
