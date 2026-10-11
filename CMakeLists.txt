cmake_minimum_required(VERSION 3.16)

project(LSH_algo LANGUAGES CXX)

add_library(lsh_core
    src/lsh_index.cpp
    src/search.cpp
    src/data_loader.cpp
)

target_include_directories(lsh_core PUBLIC include)
target_compile_features(lsh_core PUBLIC cxx_std_23)

add_executable(LSH_algo
    src/main.cpp
)

target_link_libraries(LSH_algo PRIVATE lsh_core)

enable_testing()

add_executable(LSH_tests
    tests/tests.cpp
)

target_link_libraries(LSH_tests PRIVATE lsh_core)

add_test(
    NAME LSH_tests
    COMMAND LSH_tests
)

add_executable(LSH_benchmark
    benchmark/benchmark.cpp
)
target_link_libraries(LSH_benchmark PRIVATE lsh_core)
