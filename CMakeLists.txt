#-----------------------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(flash-attention-cpp LANGUAGES CXX VERSION 1.0.0)

# host information.
message("==================================")
cmake_host_system_information(RESULT HOSTNAME QUERY HOSTNAME)
execute_process(COMMAND bash -c "lspci -nn | grep NVIDIA | wc -l" OUTPUT_VARIABLE nvidia_number)
message("hostname: ${HOSTNAME}")
message("rootpath: ${PROJECT_SOURCE_DIR}")
message("nvidia_number: ${nvidia_number}")
message("==================================")

# c++ settings.
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if (NOT ${nvidia_number} EQUAL 0)
    set(CMAKE_CXX_FLAGS "-std=c++17 -pthread -fpic -fopenmp -D_GLIBCXX_USE_CXX11_ABI=1")
else ()
    set(CMAKE_CXX_FLAGS "-std=c++17 -pthread -fpic -fopenmp -D_GLIBCXX_USE_CXX11_ABI=0")
endif ()
set(CMAKE_BUILD_TYPE Debug)

# cmake settings.
if (POLICY CMP0076)
    cmake_policy(SET CMP0076 NEW)
endif ()

# cuda settings.
if (NOT ${nvidia_number} EQUAL 0)
    set(CMAKE_CUDA_COMPILER /usr/bin/clang++)
    set(CMAKE_CUDA_ARCHITECTURES "80")
else ()
    set(CMAKE_CUDA_COMPILER /usr/local/corex/bin/clang++)
    set(CMAKE_CUDA_ARCHITECTURES "ivcore10;ivcore11;ivcore20")
endif ()
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_FLAGS "-std=c++17")
# enable after cuda compiler set.
enable_language(CUDA)
message("CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}")
message("CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

#-----------------------------------------------------------------------------------------------------------------------
# cuda.
if (NOT ${nvidia_number} EQUAL 0)
    set(CUDA_INCLUDE "/usr/local/cuda/include")
    find_library(CUDART_LIBRARY cudart /usr/local/cuda/lib64)
    # workaround for ld error when compile.
    execute_process(COMMAND bash -c "ln -sf /usr/local/cuda/lib64/libcudadevrt.a /usr/lib/libcudadevrt.a" OUTPUT_VARIABLE nvidia_number)
    execute_process(COMMAND bash -c "ln -sf /usr/local/cuda/lib64/libcudart_static.a /usr/lib/libcudart_static.a" OUTPUT_VARIABLE nvidia_number)
else ()
    set(CUDA_INCLUDE "/usr/local/corex/include")
    find_library(CUDART_LIBRARY cudart /usr/local/corex/lib64)
endif ()
message("CUDA_INCLUDE=${CUDA_INCLUDE}")
message("CUDART_LIBRARY=${CUDART_LIBRARY}")

#-----------------------------------------------------------------------------------------------------------------------
# torch.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_C10D_NCCL")
set(TORCH_INCLUDE /usr/local/torch/include)
set(TORCH_API_INCLUDE /usr/local/torch/include/torch/csrc/api/include)
find_library(TORCH_LIBRARY torch /usr/local/torch/lib)
find_library(TORCH_CPU_LIBRARY torch_cpu /usr/local/torch/lib)
find_library(TORCH_CUDA_LIBRARY torch_cuda /usr/local/torch/lib)
find_library(TORCH_C10_LIBRARY c10 /usr/local/torch/lib)
find_library(TORCH_C10_CUDA_LIBRARY c10_cuda /usr/local/torch/lib)
message("==================================")
message("TORCH_INCLUDE=${TORCH_INCLUDE}")
message("TORCH_API_INCLUDE=${TORCH_API_INCLUDE}")
message("TORCH_LIBRARY=${TORCH_LIBRARY}")
message("TORCH_CPU_LIBRARY=${TORCH_CPU_LIBRARY}")
message("TORCH_CUDA_LIBRARY=${TORCH_CUDA_LIBRARY}")
message("TORCH_C10_LIBRARY=${TORCH_C10_LIBRARY}")
message("TORCH_C10_CUDA_LIBRARY=${TORCH_C10_CUDA_LIBRARY}")
message("==================================")

#-----------------------------------------------------------------------------------------------------------------------
# sources.
file(GLOB_RECURSE TEST_SRCS src/flash-attention/*.cpp src/flash-attention/*.cu)
message("TEST_SRCS=${TEST_SRCS}")

# set target.
add_executable(runTests)
target_compile_options(runTests PUBLIC -Wno-deprecated-declarations)
target_include_directories(runTests PUBLIC
        gtest
        gtest/include
        include
        src/flash-attention
        # cuda.
        ${CUDA_INCLUDE}
        # torch.
        ${TORCH_INCLUDE}
        ${TORCH_API_INCLUDE}
)
target_sources(runTests PUBLIC
        src/main.cpp
        gtest/src/gtest-all.cc
        src/api.cpp
        ${TEST_SRCS}
)
target_link_libraries(runTests PUBLIC
        # cuda.
        ${CUDART_LIBRARY}
        # torch.
        ${TORCH_LIBRARY}
        ${TORCH_CPU_LIBRARY}
        ${TORCH_CUDA_LIBRARY}
        ${TORCH_C10_LIBRARY}
        ${TORCH_C10_CUDA_LIBRARY}
)
