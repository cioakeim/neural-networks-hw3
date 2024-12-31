find_package(OpenMP REQUIRED)
include_directories(${OpenMP_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include/CommonLib)
include_directories(${PROJECT_SOURCE_DIR}/include/MLP)


file(GLOB AENC_SOURCES "${PROJECT_SOURCE_DIR}/src/AutoEncoder/*.cpp")
add_library(aenc_lib STATIC ${AENC_SOURCES})
target_compile_options(aenc_lib PUBLIC -O3 -Wall -Wextra -fopenmp)
target_link_libraries(aenc_lib PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(aenc_lib PUBLIC common_lib)
target_link_libraries(aenc_lib PUBLIC mlp_lib)
target_link_libraries(aenc_lib PUBLIC ${AOCL_BLAS_LIB})
target_link_libraries(aenc_lib PUBLIC ${AOCL_LAPACK_LIB})
