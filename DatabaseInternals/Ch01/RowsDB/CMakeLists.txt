cmake_minimum_required(VERSION 3.16)
project(RowsDB LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(rowsdb main.cpp src/row_store.cpp)
target_include_directories(rowsdb PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(rowsdb PRIVATE -Wall -Wextra -O2)
endif()