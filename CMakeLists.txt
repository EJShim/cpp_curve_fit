project(pybind_demo)
cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)	

# only for macosx
set(CMAKE_OSX_ARCHITECTURES "x86_64")


# Add Executable
add_executable(demo 
				main.cpp
			)

add_executable(ceres_demo
				ceres_demo.cpp)

# 3rd party libraries
include(FetchContent)
FetchContent_Declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11.git
	GIT_TAG        v2.10.3
	GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(pybind11)

#Eigen
FetchContent_Declare(
	eigen
	GIT_REPOSITORY https://gitlab.com/libeigen/eigen
	GIT_TAG        9623c0c4b9f40e74ba6ba551f86831616bdd488c
)
FetchContent_MakeAvailable(eigen)

# Ceres
FetchContent_Declare(
	ceres
	GIT_REPOSITORY	https://github.com/ceres-solver/ceres-solver
	GIT_TAG			2.1.0
)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(PROVIDE_UNINSTALL_TARGET OFF)
set(MINIGLOG ON)
set(CUDA OFF)
FetchContent_MakeAvailable(ceres)


target_link_libraries(
	demo 
	PRIVATE 
	pybind11::embed		
)

target_link_libraries(
	ceres_demo
	PRIVATE
	pybind11::embed
	Ceres::ceres
)

# Copy python module
add_custom_command(
        TARGET demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/module.py
                ${CMAKE_CURRENT_BINARY_DIR}/module.py)





message("#################################")
message(${ceres_BINARY_DIR}/bin/ceres.dll)

add_custom_command(
	TARGET ceres_demo POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy
			${ceres_BINARY_DIR}/bin/ceres.dll
			${CMAKE_CURRENT_BINARY_DIR}/ceres.dll)
