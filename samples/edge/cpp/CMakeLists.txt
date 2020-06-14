cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME EdgeDetect)
project(${PROJECT_NAME})

##################################################
# set CMAKE_INSTALL_PREFIX
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR}/build CACHE PATH "..." FORCE)
endif()
set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# Disable ZERO_CHECK project
set(CMAKE_SUPPRESS_REGENERATION true)

# Default to `release` build if nothing was specified
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to release")
  set(CMAKE_BUILD_TYPE release)
endif()
##################################################

# OpenCV dependencies
find_package(OpenCV REQUIRED)

set(INCLUDE_DIRS
${OpenCV_INCLUDE_DIRS}
)
set(LIBS
${OpenCV_LIBS}
)

##################################################
# OpenCV Capture Project
add_executable(${PROJECT_NAME} src/edgedetect.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${LIBS})
##################################################




