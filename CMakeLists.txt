# Program details
cmake_minimum_required(VERSION 3.10)
project(QuickFind_exe)
set(CMAKE_CXX_STANDARD 20)

# Libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV4 REQUIRED opencv4)

# Compile the following files
add_executable(QuickFind_exe main.cpp quickfind.cpp quickfind.h)
target_link_libraries(QuickFind_exe ${OPENCV4_LIBRARIES})
target_include_directories(QuickFind_exe PUBLIC ${OPENCV4_INCLUDE_DIRS})
target_compile_options(QuickFind_exe PUBLIC ${OPENCV4_CFLAGS_OTHER})
