# set the minimum cmake version
cmake_minimum_required(VERSION 3.10)

# set the project name
project(sips LANGUAGES CXX)

# set the cpp standard
# set(CMAKE_CXX_STANDARD 11)
    # environment reason (pybind11 3.1 requires C++14)
        # 261113 YO
set(CMAKE_CXX_STANDARD 14)
    ####
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# set the cpp compilor flags
# warning flags -Wall -Wextra 
# ISO compliance -pedantic
# optimization level O3
# position-independent code for shared libray -fPIC
# for the host machine -march-native
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -O3 -fPIC -march=native -flto")
    # environment reason
        # 261113 YO
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /EHsc /bigobj /openmp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -O3 -fPIC -march=native -flto")
endif()

# find python and pybind11
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

# add the target library(~ add_library)
pybind11_add_module(sips ./src/pybind_wrapper.cpp)

# Find and link OpenMP
# see https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html
# for lnking OpenMP with modern cmake
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sips PUBLIC OpenMP::OpenMP_CXX)
endif()


