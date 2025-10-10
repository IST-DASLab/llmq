cmake_minimum_required(VERSION 3.25)

# compile for native GPU if no GPU has been specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

include(FetchContent)
project(HALO_cpp CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# As cuFile might not be supported everywhere (WSL; AMD), make it an optional depencendy
# with a fallback implementation
option(USE_CUFILE "Enable cuFile support for optimized file I/O" ON)
option(USE_NVML "Enable nvml support for detailed GPU stats reporting" ON)

FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
        EXCLUDE_FROM_ALL
)

FetchContent_Declare(
        cudnn-fe
        GIT_REPOSITORY https://github.com/NVIDIA/cudnn-frontend.git
        GIT_TAG v1.12.0
        EXCLUDE_FROM_ALL
)

FetchContent_Declare(
        cli11
        QUIET
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG v2.3.2
)

# we could use fmt only if std::format is not available, but (e.g., g++-12), but
# since it is compatible with std::format, its easier to just use it consistently.
FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt
        GIT_TAG        12.0.0
)

find_package(CUDAToolkit COMPONENTS cuBLAS REQUIRED)

FetchContent_MakeAvailable(json)
FetchContent_MakeAvailable(cudnn-fe)
FetchContent_MakeAvailable(cli11)
FetchContent_MakeAvailable(fmt)

# optional cufile
if(USE_CUFILE)
    find_package(CUDAToolkit COMPONENTS cuFile QUIET)
    if(TARGET CUDA::cuFile)
        set(CUFILE_SOURCE src/utilities/cu_file.cpp)
        set(CUFILE_LIBS CUDA::cuFile)
    else()
        set(USE_CUFILE 0)
        message(WARNING "cuFile requested but not found, disabling cuFile support")
    endif()
endif ()

if(NOT USE_CUFILE)
    set(CUFILE_SOURCE src/utilities/cu_file_fallback.cpp)
    set(CUFILE_LIBS "")
endif ()

if(USE_NVML)
    find_package(CUDAToolkit COMPONENTS nvml)
    set(NVML_SOURCE src/utilities/gpu_info_nvml.cpp)
    set(NVML_LIBS CUDA::nvml)
else()
    set(NVML_SOURCE src/utilities/gpu_info_fallback.cpp)
    set(NVML_LIBS "")
endif ()

    # allow configuring search paths from env
set(NCCL_INCLUDE_DIR $ENV{NCCL_INCLUDE_DIR} CACHE PATH "Folder containing NCCL headers")
set(NCCL_LIB_DIR $ENV{NCCL_LIB_DIR} CACHE PATH "Folder containing NCCL library")
find_path(NCCL_INCLUDE_DIRS NAMES nccl.h HINTS ${NCCL_INCLUDE_DIR} REQUIRED)
find_library(NCCL_LIBRARIES NAMES nccl HINTS ${NCCL_LIB_DIR} REQUIRED)
message(STATUS "Using nccl: ${NCCL_LIBRARIES}")
add_library(nvidia::nccl SHARED IMPORTED)
set_property(TARGET nvidia::nccl PROPERTY IMPORTED_LOCATION "${NCCL_LIBRARIES}")
target_include_directories(nvidia::nccl INTERFACE "${NCCL_INCLUDE_DIRS}")

set(MULTI_GPU_DEPS "")
set(MULTI_GPU_DEFS "")
find_package(MPI)
if (MPI_FOUND)
    list(APPEND MULTI_GPU_DEPS MPI::MPI_CXX)
    list(APPEND MULTI_GPU_DEFS USE_MPI=1)
endif ()
find_package(Threads)
if (Threads_FOUND)
    list(APPEND MULTI_GPU_DEPS Threads::Threads)
    list(APPEND MULTI_GPU_DEFS USE_THREADS=1)
else ()
    message(STATUS "Threads not found, disabling")
endif ()

find_path(CUDNN_INCLUDE_DIR cudnn.h
        HINTS $ENV{CUDNN_INCLUDE_PATH} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include)

find_library(CUDNN_LIBRARY cudnn
        HINTS $ENV{CUDNN_LIBRARY_PATH} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib64 lib)

message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
message(STATUS "cuBLAS Library: ${CUDA_cublas_LIBRARY}")
message(STATUS "cuBLASLt Library: ${CUDA_cublasLt_LIBRARY}")

add_library(libllmq SHARED
        src/utilities/safetensors.cpp
        ${CUFILE_SOURCE}
        ${NVML_SOURCE}
        src/utilities/allocator.cpp
        src/utilities/utils.cpp
        src/utilities/comm.cpp
        src/utilities/tensor.cpp
        src/utilities/sol.cpp
        src/utilities/dtype.cpp

        src/training/dataloader.cpp
        src/training/logging.cpp
        src/training/checkpoint.cpp

        src/models/llama_run_state.cpp
        src/models/llama_model.cpp
        src/models/llama_weights.cpp
        src/models/llama_gradients.cpp
        src/models/llama_config.cpp

        src/kernels/kernels.cpp
        src/kernels/encoder.cu
        src/kernels/matmul.cu
        src/kernels/swiglu.cu
        src/kernels/rope.cu
        src/kernels/rmsnorm.cu
        src/kernels/cudnn_att.cpp
        src/kernels/quant.cu
        src/kernels/global_norm.cu
        src/kernels/adamw.cu
        src/kernels/attention.cu
        src/kernels/fused_classifier.cu
        src/kernels/transpose.cu
        src/kernels/vector_add.cu
        src/kernels/random.cu
        src/kernels/fill.cu
        src/kernels/convert.cu
)

target_link_libraries(libllmq PRIVATE ${CUFILE_LIBS} cudnn_frontend ${CUDNN_LIBRARY} nvidia::nccl fmt::fmt-header-only)
target_link_libraries(libllmq PUBLIC CUDA::cudart CUDA::cublasLt ${NVML_LIBS} nlohmann_json::nlohmann_json)
target_link_libraries(libllmq PUBLIC ${MULTI_GPU_DEPS})
target_compile_definitions(libllmq PUBLIC ${MULTI_GPU_DEFS})
target_include_directories(libllmq PRIVATE ${CUDNN_INCLUDE_DIR})
target_include_directories(libllmq PUBLIC src)
target_compile_options(libllmq PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(train train.cpp)
target_link_libraries(train PRIVATE libllmq CLI11::CLI11)

add_executable(recompute-test src/testing/recompute.cpp)
target_link_libraries(recompute-test PRIVATE libllmq CLI11::CLI11)

add_executable(export-checkpoint export-checkpoint.cpp)
target_link_libraries(export-checkpoint PRIVATE libllmq CLI11::CLI11)

install(TARGETS libllmq train recompute-test export-checkpoint
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)
