cmake_minimum_required(VERSION 2.8)
project(test)

find_package(OpenCV REQUIRED)

if(NOT WIN32)
ADD_DEFINITIONS("-std=c++14 -O0")
endif(NOT WIN32)

link_libraries(stdc++fs)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(include)
add_executable( KCF src/fhog.cpp src/kcftracker.cpp include/run_instance.h src/main.cpp)
add_executable( run_tracker include/fhog.hpp src/fhog.cpp src/kcftracker.cpp src/runtracker.cpp)
target_link_libraries( KCF ${OpenCV_LIBS})
target_link_libraries( run_tracker ${OpenCV_LIBS})




