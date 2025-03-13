cmake_minimum_required(VERSION 3.0)
project(megagigachad LANGUAGES CXX)

set(HEADERS A/index.h C/main.h)
set(SOURCES C/main.cpp)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib")

execute_process(COMMAND python3 ${CMAKE_SOURCE_DIR}/A/preparing.py WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/A)

add_subdirectory(B)
add_subdirectory(C)
add_executable(main ${SOURCES} ${HEADERS})
target_link_libraries(main megagigachad_library)

