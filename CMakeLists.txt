cmake_minimum_required(VERSION 3.9)
project(TheDreamMachine)

set(CMAKE_C_STANDARD 99)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (COMPILER_SUPPORTS_MARCH_NATIVE)
    list(APPEND CMAKE_C_FLAGS "-march=native")
endif ()

add_subdirectory(src)
add_subdirectory(test)
