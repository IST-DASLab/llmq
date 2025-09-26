cmake_minimum_required(VERSION 3.28)
include(FetchContent)
project(HALO_cpp CUDA CXX)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

find_package(CUDAToolkit COMPONENTS cuFile cuBLAS REQUIRED)

FetchContent_MakeAvailable(json)
FetchContent_MakeAvailable(cudnn-fe)
FetchContent_MakeAvailable(cli11)

#find_package(nlohmann_json 3.12.0 REQUIRED)

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
find_package(MPI REQUIRED)
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
        src/utilities/cu_file.cu
        src/utilities/allocator.cpp
        src/utilities/nvml.cpp
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
        src/testing/check-tensors.cu
        src/kernels/transpose.cu
        src/kernels/vector_add.cu
        src/kernels/random.cu
)

target_link_libraries(libllmq PRIVATE CUDA::cuFile cudnn_frontend ${CUDNN_LIBRARY} nvidia::nccl)
target_link_libraries(libllmq PUBLIC CUDA::cudart CUDA::cublasLt CUDA::nvml nlohmann_json::nlohmann_json)
target_link_libraries(libllmq PUBLIC ${MULTI_GPU_DEPS})
target_compile_definitions(libllmq PUBLIC ${MULTI_GPU_DEFS})
target_include_directories(libllmq PRIVATE ${CUDNN_INCLUDE_DIR})
target_include_directories(libllmq PUBLIC src)
target_compile_options(libllmq PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(train train.cpp)
target_link_libraries(train PRIVATE libllmq CLI11::CLI11)

add_executable(kernel-tests src/testing/kernel-tests.cu src/testing/reference-results.cpp)
target_link_libraries(kernel-tests PRIVATE libllmq Catch2::Catch2WithMain)

add_executable(recompute-test src/testing/recompute.cpp)
target_link_libraries(recompute-test PRIVATE libllmq CLI11::CLI11)

add_executable(export-checkpoint export-checkpoint.cpp)
target_link_libraries(export-checkpoint PRIVATE libllmq CLI11::CLI11)
