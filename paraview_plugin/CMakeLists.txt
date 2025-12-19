cmake_minimum_required(VERSION 3.15)
project(time_index LANGUAGES CXX)

# Add your local pybind11
add_subdirectory(./pybind11 pybind11_build)

# Build the module
pybind11_add_module(time_index time_index.cpp)

set_target_properties(time_index PROPERTIES
    PREFIX ""
    CXX_STANDARD 17
)
