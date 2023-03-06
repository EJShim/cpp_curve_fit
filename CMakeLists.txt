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

include(FetchContent)
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE)
set(EIGEN_BUILD_DOC OFF)
# note: To disable eigen tests,
# you should put this code in a add_subdirectory to avoid to change
# BUILD_TESTING for your own project too since variables are directory
# scoped
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
set( OFF)
FetchContent_MakeAvailable(Eigen)

# GLOG
FetchContent_Declare(
	glog
	GIT_REPOSITORY https://github.com/google/glog
	GIT_TAG        v0.6.0   
)
set(WITH_UNWIND OFF)
set(WITH_GTEST OFF)
set(WITH_GFLAGS OFF)
FetchContent_MakeAvailable(glog)

# # GFLAGS
# FetchContent_Declare(
# 	gflags
# 	GIT_REPOSITORY https://github.com/gflags/gflags
# 	GIT_TAG        v2.2.2
# )
# FetchContent_MakeAvailable(gflags)


# Ceres
FetchContent_Declare(
	Ceres
	GIT_REPOSITORY 	https://ceres-solver.googlesource.com/ceres-solver
	GIT_TAG			2.1.0
	GIT_SHALLOW		TRUE
)
FetchContent_MakeAvailable(Ceres)


target_link_libraries(
	demo 
	PRIVATE 
	pybind11::embed		
)

target_link_libraries(
	ceres_demo
	PRIVATE
	Eigen3::Eigen
	Ceres::ceres
)

# Copy python module
add_custom_command(
        TARGET demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/module.py
                ${CMAKE_CURRENT_BINARY_DIR}/module.py)

