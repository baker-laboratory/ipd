cmake_minimum_required(VERSION 3.15)
project(evn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 REQUIRED)
include_directories(${PROJECT_SOURCE_DIR})

pybind11_add_module(_token_column_format MODULE evn/format/_token_column_format.cpp)
set_target_properties(_token_column_format PROPERTIES PREFIX "" OUTPUT_NAME "_token_column_format" )
target_link_libraries(_token_column_format PRIVATE pybind11::module)
install(TARGETS _token_column_format; DESTINATION evn/format)

pybind11_add_module(_detect_formatted_blocks MODULE evn/format/_detect_formatted_blocks.cpp)
set_target_properties(_detect_formatted_blocks PROPERTIES PREFIX "" OUTPUT_NAME "_detect_formatted_blocks" )
target_link_libraries(_detect_formatted_blocks PRIVATE pybind11::module)
install(TARGETS _detect_formatted_blocks; DESTINATION evn/format)
