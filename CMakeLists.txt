project(pybind_demo)
cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 14)	

# only for macosx
set(CMAKE_OSX_ARCHITECTURES "x86_64")


# Add Executable
add_executable(demo 
				main.cpp
			)

# 3rd party libraries
include(FetchContent)
FetchContent_Declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11.git
	GIT_TAG        v2.10.3
	GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(pybind11)


target_link_libraries(
	demo 
	PRIVATE 
	pybind11::embed		
)

# Copy python module
add_custom_command(
        TARGET demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/module.py
                ${CMAKE_CURRENT_BINARY_DIR}/module.py)

