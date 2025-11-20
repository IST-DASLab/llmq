cmake_minimum_required(VERSION 3.25)

# compile for native GPU if no GPU has been specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

include(FetchContent)
project(LLMQ CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# As cuFile might not be supported everywhere (WSL; AMD), make it an optional dependency
# with a fallback implementation
option(USE_CUFILE "Enable cuFile support for optimized file I/O" ON)
option(USE_NVML "Enable nvml support for detailed GPU stats reporting" ON)
option(USE_MPI "Enable MPI backend for multi-GPU support" ON)
option(PYTHON_BINDING "Build python bindings" OFF)

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
        GIT_TAG v2.6.1
)
FetchContent_Declare(
        nanobind
        QUIET
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG v2.9.2
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
if (USE_MPI)
    find_package(MPI REQUIRED)
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

add_library(llmq-common SHARED
        src/utilities/safetensors.cpp
        ${CUFILE_SOURCE}
        ${NVML_SOURCE}
        src/utilities/allocator.cpp
        src/utilities/utils.cpp
        src/utilities/comm.cpp
        src/utilities/tensor.cpp
        src/utilities/sol.cpp
        src/utilities/dtype.cpp
        src/utilities/stack.cpp

        src/training/dataloader.cpp
        src/training/logging.cpp
        src/training/checkpoint.cpp

        src/models/llama_run_state.cpp
        src/models/llama_model.cpp
        src/models/llama_weights.cpp
        src/models/llama_gradients.cpp
        src/models/llama_config.cpp
        src/models/llama_optimizer.cpp

        src/kernels/kernels.cpp
        src/kernels/encoder.cu
        src/kernels/matmul.cpp
        src/kernels/swiglu.cu
        src/kernels/rope.cu
        src/kernels/rmsnorm.cu
        src/kernels/cudnn_att.cpp
        src/kernels/quant.cu
        src/kernels/global_norm.cu
        src/kernels/bias.cu
        src/kernels/adamw.cu
        src/kernels/attention.cu
        src/kernels/fused_classifier.cu
        src/kernels/transpose.cu
        src/kernels/vector_add.cu
        src/kernels/random.cu
        src/kernels/fill.cu
        src/kernels/convert.cu
)

target_link_libraries(llmq-common PRIVATE ${CUFILE_LIBS} cudnn_frontend ${CUDNN_LIBRARY} nvidia::nccl fmt::fmt-header-only)
target_link_libraries(llmq-common PUBLIC CUDA::cudart CUDA::cublasLt ${NVML_LIBS} nlohmann_json::nlohmann_json)
target_link_libraries(llmq-common PUBLIC ${MULTI_GPU_DEPS})
target_compile_definitions(llmq-common PUBLIC ${MULTI_GPU_DEFS})
target_include_directories(llmq-common PRIVATE ${CUDNN_INCLUDE_DIR})
target_include_directories(llmq-common PUBLIC src)
target_compile_options(llmq-common PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(train train.cpp)
target_link_libraries(train PRIVATE llmq-common CLI11::CLI11)

add_executable(export-checkpoint export-checkpoint.cpp)
target_link_libraries(export-checkpoint PRIVATE llmq-common CLI11::CLI11)

if (NOT SKBUILD)
    install(TARGETS llmq-common train export-checkpoint
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin
    )
endif ()


if (PYTHON_BINDING)
    find_package(Python COMPONENTS Interpreter Development.Module OPTIONAL_COMPONENTS Development.SABIModule)
    FetchContent_MakeAvailable(nanobind)
    nanobind_add_module(_pyllmq STABLE_ABI src/binding/binding.cpp src/binding/py_train.cpp)
    target_link_libraries(_pyllmq PUBLIC llmq-common Python::Module fmt::fmt-header-only)

    # make sure we pick up cuda libraries from within the current python env, if available
    set(RPATH_LIST
        "$ORIGIN"
        "$ORIGIN/../nvidia/nccl/lib"
        "$ORIGIN/../nvidia/cudnn/lib"
        "$ORIGIN/../nvidia/cuda_runtime/lib"
        "$ORIGIN/../nvidia/cublas/lib"
    )
    list(JOIN RPATH_LIST ":" WHEEL_RPATH)

    set_target_properties(_pyllmq llmq-common PROPERTIES
            INSTALL_RPATH "${WHEEL_RPATH}"
            BUILD_WITH_INSTALL_RPATH OFF
            INSTALL_RPATH_USE_LINK_PATH FALSE
    )

    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/binding/python/
            DESTINATION pyllmq
    )

    install(TARGETS _pyllmq llmq-common
            LIBRARY DESTINATION pyllmq
    )

    # Install scripts as a subpackage
    install(DIRECTORY scripts/
            DESTINATION pyllmq/scripts
            FILES_MATCHING PATTERN "*.py"
    )
endif ()
