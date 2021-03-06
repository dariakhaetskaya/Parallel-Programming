cmake_minimum_required(VERSION 3.16)
project(SLE_Solver)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

set(SOURCE_FILES ParallelSolver.cpp)

add_executable(untitled8 ParallelSolver.cpp)