cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(gcn)

SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

#list(APPEND CMAKE_MODULE_PATH "/usr/local/libtorch/share/cmake/Torch/TorchConfig.cmake")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/local/libtorch/")
find_package(Torch REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#add_executable(layer layer.cpp)
set(EXECUTABLE_NAME gcn)
add_executable(${EXECUTABLE_NAME})

target_sources(${EXECUTABLE_NAME} PRIVATE 
    src/model.cpp
    src/layer.cpp
    include/layer.h
)

target_include_directories(${EXECUTABLE_NAME} PRIVATE include)

target_link_libraries(${EXECUTABLE_NAME} "${TORCH_LIBRARIES}")

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

