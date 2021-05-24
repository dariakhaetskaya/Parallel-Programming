cmake_minimum_required(VERSION 3.16)
project(Lab5Cluster)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(SOURCES cluster.cpp)

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

add_executable(Lab5Cluster ${SOURCES})
