cmake_minimum_required(VERSION 2.8) 
PROJECT(Towel_detection)
find_package(OpenCV REQUIRED)

set(NAME_SRC
    src/Towel_detection.cpp
    src/towel_test.cpp

)

set(NAME_HEADERS       
    include/Towel_detection.hpp
    include/Eserial.h
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories(${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable(Towel_detection ${NAME_SRC} ${NAME_HEADERS})

target_link_libraries(Towel_detection ${OpenCV_LIBS})
