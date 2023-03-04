project(pybind_demo)
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 14)	

if(APPLE)
	set(CMAKE_OSX_ARCHITECTURES "x86_64")
endif(APPLE)

include(FetchContent)
FetchContent_Declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11.git
	GIT_TAG        v2.6.2
	GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(pybind11)


# Add Executable
add_executable(demo main.cpp)
target_link_libraries(demo PRIVATE pybind11::embed)

# Copy python module
add_custom_command(
        TARGET demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/module.py
                ${CMAKE_CURRENT_BINARY_DIR}/module.py)