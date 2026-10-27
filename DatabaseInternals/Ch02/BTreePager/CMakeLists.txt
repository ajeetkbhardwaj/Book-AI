cmake_minimum_required(VERSION 3.16)
project(BTreePager LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)

add_executable(btree_pager 
  main.cpp 
  src/page_manager.cpp 
  src/btree.cpp
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(btree_pager PRIVATE -Wall -Wextra -O2)
endif()
