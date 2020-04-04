cmake_minimum_required(VERSION 2.8.12)
project(smallpt_pybind)

add_subdirectory(pybind11)
pybind11_add_module(smallpt_pybind smallpt_pybind.cpp)