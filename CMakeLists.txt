cmake_minimum_required(VERSION 3.8)

project(ml)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTS "Build tests using Catch2" OFF)



# ENCS Eigen3 CMake setup
if (WIN32)
    include_directories(C:/eigen-3.4.1/)
else() 
    # 1. Check for our Docker environment first
    if(EXISTS "/usr/local/include/Eigen")
        message(STATUS "Docker environment detected. Using manual Eigen path.")
        include_directories("/usr/local/include/Eigen")
        # We define this so the code below knows we found it
        set(EIGEN_FOUND_MANUALLY TRUE)
    endif()

    # 2. If we didn't find it in Docker, try the standard find_package (for School/Mac)
    if(NOT EIGEN_FOUND_MANUALLY)
        find_package(Eigen3 REQUIRED)
        include_directories(${EIGEN3_INCLUDE_DIRS})
        link_libraries(Eigen3::Eigen)
    endif()
endif()

include_directories(src/)
aux_source_directory(src SOURCE)

if(BUILD_TESTS)
    enable_testing()
    find_package(Catch2 3 REQUIRED)
    add_executable(ml tests/main.cpp ${SOURCE})
    target_link_libraries(ml PRIVATE Catch2::Catch2WithMain)
    include(CTest)
    include(Catch)
    catch_discover_tests(ml)
else()
    add_executable(ml main.cpp ${SOURCE})
endif()




