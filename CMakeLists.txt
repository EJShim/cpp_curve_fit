project(pybind_demo)
cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)	

# only for macosx
set(CMAKE_OSX_ARCHITECTURES "x86_64")


# Add Executable
add_executable(demo 
				main.cpp
				curve_fit.cpp)

				include(FetchContent)
FetchContent_Declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11.git
	GIT_TAG        v2.6.2
	GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(pybind11)

FetchContent_Declare(
	GSL
	GIT_REPOSITORY 	https://github.com/ampl/gsl.git
	GIT_TAG			v2.7.0
	GIT_SHALLOW		FALSE
)
FetchContent_MakeAvailable(GSL)
include_directories(${GSL_BINARY_DIR})

target_link_libraries(
	demo 
	PRIVATE 
	pybind11::embed	
	gsl
)

# Copy python module
add_custom_command(
        TARGET demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/module.py
                ${CMAKE_CURRENT_BINARY_DIR}/module.py)