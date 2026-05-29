cmake_minimum_required(VERSION 3.8)

project(ml)

set(CMAKE_CXX_STANDARD 14)

if (EXISTS "/usr/local/include/Eigen")
    message(STATUS "Docker environment detected. Using manual Eigen path.")
    include_directories("/usr/local/include/Eigen")
    set(EIGEN_FOUND_MANUALLY TRUE)

endif()

if (NOT EIGEN_FOUND_MANUALLY)
    find_package(Eigen3 REQUIRED)
    include_directories(${EIGEN3_INCLUDE_DIRS})
    link_libraries(Eigen3::Eigen)
endif()

include_directories(src/)
aux_source_directory(src SOURCE)

add_executable(ml main.cpp ${SOURCE})
