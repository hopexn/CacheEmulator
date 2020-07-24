cmake_minimum_required(VERSION 3.14)
project(LibCache)

set(CMAKE_CXX_COMPILER g++)

find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 17)

add_executable(test_cache_emu test.cpp apis.cpp test.cpp cache.hpp request.hpp cache_emu.hpp feature.hpp)