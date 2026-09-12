cmake_minimum_required(VERSION 3.8)

project(ml)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
# if it detects apple then its on the local machine so we can use apple's special hardware to run it faster

message(STATUS "APPLE DETECTED - Optimizing build with Accelerate framework")

    # Disable tests & docs BEFORE fetching
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)

    include(FetchContent)
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE # Speeds up downloading
    )
    FetchContent_MakeAvailable(Eigen)

    # Alias target for Eigen 3.4.0 FetchContent compatibility
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen ALIAS eigen)
    endif()

    # Collect source files (explicit globbing or manual list is safer)
    file(GLOB_RECURSE SOURCES "src/*.cpp")

    add_executable(ml main.cpp ${SOURCES})

    # Header search paths for target
    target_include_directories(ml PRIVATE src/)

    # Link Eigen
    target_link_libraries(ml PRIVATE Eigen3::Eigen)

    # Enable Apple Accelerate (BLAS backend
    target_compile_definitions(ml PRIVATE EIGEN_USE_BLAS)

    # Link Frameworks & Compiler Flags
    target_link_libraries(ml PRIVATE "-framework Accelerate")
    target_compile_options(ml PRIVATE -O3 -mcpu=native)

else()
# if it is not on apple, then it's in the docker container, which is basically the same code as before. 
message(STATUS "APPLE NOT DETECTED")

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
    aux_source_directory(tests/ TEST_SOURCES)
    add_executable(ml ${SOURCE} ${TEST_SOURCES})
    target_link_libraries(ml PRIVATE Catch2::Catch2WithMain)
    include(CTest)
    include(Catch)
    catch_discover_tests(ml)
else()
    add_executable(ml main.cpp ${SOURCE})
endif()

endif()