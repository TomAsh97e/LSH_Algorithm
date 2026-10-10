cmake_minimum_required(VERSION 3.16)

project(LSH_algo LANGUAGES CXX)

add_executable(LSH_algo
    src/main.cpp
    src/lsh_index.cpp
    src/search.cpp
)

target_include_directories(LSH_algo PRIVATE include)
target_compile_features(LSH_algo PRIVATE cxx_std_23)
