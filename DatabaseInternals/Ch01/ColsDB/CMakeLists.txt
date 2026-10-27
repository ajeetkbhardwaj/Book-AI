cmake_minimum_required(VERSION 3.16)
project(ColsDB LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(colsdb main.cpp src/col_store.cpp)

# Tell CMake where to find our header files
target_include_directories(colsdb PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(colsdb PRIVATE -Wall -Wextra -O2)
endif()