cmake_minimum_required(VERSION 3.7.2 FATAL_ERROR)

project(gpucomp C)

set(CMAKE_C_COMPILER "gcc")

set(CMAKE_C_STANDARD 11)

include(GNUInstallDirs)

find_package(OpenCL REQUIRED)

add_library(opencl INTERFACE IMPORTED)

set_property(TARGET opencl PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES ${OpenCL_INCLUDE_DIRS})

set_property(TARGET opencl PROPERTY
        INTERFACE_LINK_LIBRARIES ${OpenCL_LIBRARY})

link_libraries(opencl)
link_libraries(m)

add_executable(gpucomp main.c opencl.c ex01.c ex02.c ex03.c ex04.c ex05.c ex06.c)
