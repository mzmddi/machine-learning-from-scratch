cmake_minimum_required(VERSION 3.8)

project(ml)

set(CMAKE_CXX_STANDARD 14)

include_directories(src/)
aux_source_directory(src SOURCE)

add_executable(ml main.cpp ${SOURCE})
