cmake_minimum_required(VERSION 3.5)
project(NSAP)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES SourceFiles/main.cpp SourceFiles/CliqueFinder.cpp Headers/CliqueFinder.h SourceFiles/Organism.cpp Headers/Organism.h SourceFiles/Graph.cpp Headers/Graph.h Headers/Edge.h SourceFiles/Vertex.cpp Headers/Vertex.h)
add_executable(NSAP ${SOURCE_FILES})