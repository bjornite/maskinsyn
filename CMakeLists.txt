cmake_minimum_required(VERSION 3.5)
project(Maskinsyn)

find_package(OpenCV 3 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp Moving_object_tracker.cpp Moving_object_tracker.h)
add_executable(Maskinsyn ${SOURCE_FILES})
include_directories($(OPENCV_INCLUDE_DIRS))

target_link_libraries( Maskinsyn
        ${OpenCV_LIBS}
        )
