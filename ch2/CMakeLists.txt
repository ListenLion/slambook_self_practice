#
cmake_minimum_required(VERSION 2.8)
project(Helloslam)


include_directories(${PROJECT_SOURCE_DIR}/inlcude)
set( CMAKE_BUILD_TYPE "DEBUG")



add_library(hello_shared SHARED lib/libhelloslam.cpp)

add_executable(hellslam src/helloslam.cpp)

add_executable(usehello src/usehello.cpp)
target_link_libraries(usehello hello_shared)


