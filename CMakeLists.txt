cmake_minimum_required(VERSION 3.20)

project(cactus-gaussian LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CUDA_STANDARD 17)
#set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

# set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
set(CMAKE_CUDA_ARCHITECTURES 89 90 100 101 120)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

# 设置MSVC编译编码,下面需要设置NVCC正确转发MSVC风格的编译选项
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/source-charset:utf-8>")


add_library(gaussian_lib STATIC gaussian.h gaussian_cpu.cpp gaussian_gpu.cu common.h)


target_compile_options(gaussian_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -forward-slash-prefix-opts -forward-unknown-opts>)
target_compile_features(gaussian_lib PRIVATE cxx_std_17 cuda_std_17)

# 编译静态库需要打开这两个选项
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_RESOLVE_DEVICE_SYMBOLS.html
set_target_properties(gaussian_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gaussian_lib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)



target_include_directories(gaussian_lib PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)


message(STATUS "CUDA HDR: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

add_executable(gaussian main.cpp gaussian.h)
target_compile_features(gaussian PRIVATE cxx_std_17)
target_link_libraries(gaussian gaussian_lib)



