cmake_minimum_required(VERSION 3.16)
project(AppendOnlyHeapStore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)

add_executable(heap_store 
  main.cpp 
  src/append_only_heap_store.cpp
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(heap_store PRIVATE -Wall -Wextra -O2)
endif()
