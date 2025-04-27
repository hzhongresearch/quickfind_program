# Program details
cmake_minimum_required(VERSION 3.10)
project(QuickFind_exe)
set(CMAKE_CXX_STANDARD 20)

# Libraries
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Compile the following files
add_executable(QuickFind_exe main.cpp quickfind.cpp quickfind.h)
target_link_libraries(QuickFind_exe ${OpenCV_LIBS})
