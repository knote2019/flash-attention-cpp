cmake_minimum_required(VERSION 3.16)

project(flash_attention_cpp LANGUAGES C CXX CUDA)

if (POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif ()

find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif (CCACHE_FOUND)

set(CMAKE_VERBOSE_MAKEFILE ${FAI_VERBOSE_MAKEFILE})

set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")

set(CMAKE_EXE_LINKER_FLAGS "-Wl,--as-needed")
set(CMAKE_CUDA_ARCHITECTURES "80")

add_compile_options(
        -Wall
        -Werror
        -Wextra
        # -Wswitch-default
        # -Wfloat-equal
        -Wshadow
        -Wcast-qual
        -Wno-strict-aliasing
)

# Nvidia GPU
find_package(CUDA REQUIRED)
# unset (CUDA_USE_STATIC_CUDA_RUNTIME CACHE)
# option (CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set(CUDA_VERBOSE_BUILD ${FAI_VERBOSE_MAKEFILE})
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 -Xcompiler -fopenmp --expt-relaxed-constexpr")
if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_${CMAKE_CUDA_ARCHITECTURES} -g -lineinfo -Xptxas=-v -O0")
else ()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES} --use_fast_math -O3")
endif ()


find_package(gflags REQUIRED)

include_directories(
        include
        /usr/local/cuda/include
        ${GFLAGS_INCLUDE_DIR}
)

link_directories(
        /usr/local/cuda/lib64
)

file(GLOB FAI_SRCS
        src/main.cpp
        src/ops/*.cpp
        src/flash_attn_v2/*.cu
)
cuda_add_executable(runTests ${FAI_SRCS})
target_link_libraries(runTests ${GFLAGS_LIBRARIES})
