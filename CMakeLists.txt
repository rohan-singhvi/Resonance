cmake_minimum_required(VERSION 3.18)
project(AcousticSimulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(TBB REQUIRED)

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(FFTW3 fftw3f) 
endif()

include(CheckLanguage)
check_language(CUDA)

set(GPU_HARDWARE_AVAILABLE OFF)
find_program(NVIDIA_SMI "nvidia-smi")
if(NVIDIA_SMI)
    execute_process(COMMAND ${NVIDIA_SMI} -L 
        RESULT_VARIABLE SMI_RET 
        OUTPUT_QUIET ERROR_QUIET)
    if(SMI_RET EQUAL 0)
        set(GPU_HARDWARE_AVAILABLE ON)
    endif()
endif()

if(CMAKE_CUDA_COMPILER AND GPU_HARDWARE_AVAILABLE)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()

    set(ENABLE_CUDA ON)
    message(STATUS "CUDA Found & GPU Hardware Detected! Building Hybrid GPU/CPU version.")
else()
    set(ENABLE_CUDA OFF)
    if(CMAKE_CUDA_COMPILER)
        message(STATUS "NVCC found, but GPU hardware check failed (e.g. Mac/CPU Host). Disabling CUDA.")
    else()
        message(STATUS "CUDA NOT Found. Building CPU-only version.")
    endif()
    
    if(NOT FFTW3_FOUND)
        message(WARNING "FFTW3 not found. CPU Convolution will be slow.")
    endif()
endif()

# --- Metal detection (macOS only, no full Xcode required) ---
# We use runtime shader compilation (newLibraryWithSource:) so we only need
# the Metal.framework headers, not the xcrun metal compiler toolchain.
set(ENABLE_METAL OFF)
if(APPLE AND NOT ENABLE_CUDA)
    # Metal.framework ships with the macOS SDK; probe for its header.
    find_path(METAL_INCLUDE_DIR Metal/Metal.h
        PATHS /System/Library/Frameworks/Metal.framework/Headers
              /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Metal.framework/Headers)
    if(METAL_INCLUDE_DIR OR EXISTS "/System/Library/Frameworks/Metal.framework")
        set(ENABLE_METAL ON)
        message(STATUS "Metal framework found. Building Metal GPU backend.")
    else()
        message(STATUS "Metal framework not found. Building CPU-only version.")
    endif()
endif()

# --- Sources ---
set(SOURCES
    src/main.cpp
    src/cpu/simulation_cpu.cpp
    src/cpu/convolution_cpu.cpp
)

if(ENABLE_CUDA)
    list(APPEND SOURCES
        src/gpu/simulation_gpu.cu
        src/gpu/convolution_gpu.cu
    )
endif()

if(ENABLE_METAL)
    list(APPEND SOURCES src/metal/simulation_metal.mm)
    # .mm files need Objective-C++ compilation; set_source_files_properties is
    # the standard way to tell CMake to treat them as ObjC++.
    set_source_files_properties(src/metal/simulation_metal.mm
        PROPERTIES COMPILE_FLAGS "-x objective-c++")
endif()

add_executable(acoustic_sim ${SOURCES})

# --- Compile Definitions and Links ---
target_link_libraries(acoustic_sim PRIVATE TBB::tbb)

if(ENABLE_CUDA)
    target_compile_definitions(acoustic_sim PRIVATE ENABLE_CUDA=1)
    target_link_libraries(acoustic_sim PRIVATE CUDA::cufft CUDA::cudart)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(acoustic_sim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
    endif()
endif()

if(ENABLE_METAL)
    target_compile_definitions(acoustic_sim PRIVATE ENABLE_METAL=1)
    target_link_libraries(acoustic_sim PRIVATE
        "-framework Metal"
        "-framework Foundation"
    )
    target_include_directories(acoustic_sim PRIVATE src/metal)
endif()

target_include_directories(acoustic_sim PRIVATE include src/gpu)

if(FFTW3_FOUND)
    target_include_directories(acoustic_sim PRIVATE ${FFTW3_INCLUDE_DIRS})
    target_link_libraries(acoustic_sim PRIVATE ${FFTW3_LIBRARIES})
    target_compile_definitions(acoustic_sim PRIVATE USE_FFTW)
endif()