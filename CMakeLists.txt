cmake_minimum_required(VERSION 3.10)

project(NeuralNetworksProject2 VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


include_directories(${PROJECT_SOURCE_DIR}/include)

# Define paths for AOCL
set(HOME_DIR "/home/chris" CACHE PATH "Home Dir")
set(AOCL_PATH "${HOME_DIR}/aocl/5.0.0/aocc/lib")

set(AOCL_BLAS_LIB "${AOCL_PATH}/libblis-mt.so" CACHE PATH "BLAS")
set(AOCL_LAPACK_LIB "${AOCL_PATH}/libflame.so" CACHE PATH "LAPACK")

add_subdirectory("${PROJECT_SOURCE_DIR}/src/CommonLib")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/MLP")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/AutoEncoder")

# Standard CMake Structure 

include_directories(${MY_HOME_DIR}/my_libroot/include)

# Scripts folder
set(TESTS "${PROJECT_SOURCE_DIR}/test")



add_executable(dummy "${TESTS}/dummyTest.cpp")
target_link_libraries(dummy PRIVATE common_lib)
target_link_libraries(dummy PRIVATE mlp_lib)
target_link_libraries(dummy PRIVATE aenc_lib)
target_compile_options(dummy PRIVATE -O3 -Wall -Wextra)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)



