cmake_minimum_required(VERSION 3.16)
project(DBMS_Pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)

add_executable(dbms_pipeline main.cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(dbms_pipeline PRIVATE -Wall -Wextra -O2)
endif()
