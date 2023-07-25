cmake_minimum_required(VERSION 3.21)
project(ARBlockT3)

ENABLE_LANGUAGE(CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb3 -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -Wall")

FIND_PACKAGE(OpenCV REQUIRED)
LINK_LIBRARIES(${OpenCV_LIBS})
include_directories("${OpenCV_INCLUDE_DIRS}")


add_executable(ARBlockT3 locateobj.cpp rva.cpp rva.h)

