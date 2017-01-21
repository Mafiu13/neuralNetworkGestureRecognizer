cmake_minimum_required(VERSION 2.8)

project(siecneuronowa)

include_directories(include)

add_executable(siec
	src/main.cpp
        include/neuralnetwork.hpp
	src/neuralnetwork.cpp
)
