cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
project("ggml-easy" C CXX) #Include both C and C++
include(CheckIncludeFileCXX) 

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) # Make sure the output folder is /bin
add_subdirectory(ggml) # Add files in ggml/ folder, which is where ggml.h exists

include_directories(${CMAKE_CURRENT_SOURCE_DIR}) # Include current directory - for ggml-easy.h
# demo
set(DEMO_TARGETS
    basic-matmul-f16
    basic-matmul-f32
    basic-matmul-q4
    basic-loop
    dyt-rms
    svd
    kyutai-mimi
    safetensors
    basic-sum
    llama3.2
    ) # Target all the example files

foreach(TARGET ${DEMO_TARGETS}) # For each example file
  add_executable(${TARGET} demo/${TARGET}.cpp) # Compile to the demo/file.cpp
  target_link_libraries(${TARGET} PRIVATE ggml) # Include ggml library
  target_compile_features(${TARGET} PRIVATE cxx_std_17) # Requires C++ v17 standard, whatever that means, I'm sure it's importants
endforeach()
