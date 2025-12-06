# This file sets up
find_package(CUDAToolkit COMPONENTS cuFile cuBLAS REQUIRED)

FetchContent_MakeAvailable(cudnn-fe)

# allow configuring search paths from env
set(NCCL_INCLUDE_DIR $ENV{NCCL_INCLUDE_DIR} CACHE PATH "Folder containing NCCL headers")
set(NCCL_LIB_DIR $ENV{NCCL_LIB_DIR} CACHE PATH "Folder containing NCCL library")
find_path(NCCL_INCLUDE_DIRS NAMES nccl.h HINTS ${NCCL_INCLUDE_DIR} REQUIRED)
find_library(NCCL_LIBRARIES NAMES nccl HINTS ${NCCL_LIB_DIR} REQUIRED)
message(STATUS "Using nccl: ${NCCL_LIBRARIES}")
add_library(nvidia::nccl SHARED IMPORTED)
set_property(TARGET nvidia::nccl PROPERTY IMPORTED_LOCATION "${NCCL_LIBRARIES}")
target_include_directories(nvidia::nccl INTERFACE "${NCCL_INCLUDE_DIRS}")

if(USE_NVML)
    find_package(CUDAToolkit COMPONENTS nvml)
    set(NVML_SOURCE src/utilities/gpu_info_nvml.cpp)
    set(NVML_LIBS CUDA::nvml)
else()
    set(NVML_SOURCE src/utilities/gpu_info_fallback.cpp)
    set(NVML_LIBS "")
endif ()

find_path(CUDNN_INCLUDE_DIR cudnn.h
        HINTS $ENV{CUDNN_INCLUDE_PATH} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include)

find_library(CUDNN_LIBRARY cudnn
        HINTS $ENV{CUDNN_LIBRARY_PATH} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib64 lib)

add_library(nvidia_cudnn INTERFACE)
target_link_libraries(nvidia_cudnn INTERFACE ${CUDNN_LIBRARY})
target_include_directories(nvidia_cudnn INTERFACE ${CUDNN_INCLUDE_DIR})
add_library(nvidia::cudnn ALIAS nvidia_cudnn)

set(PRIVATE_GPU_LIBS CUDA::cuFile cudnn_frontend nvidia::cudnn nvidia::nccl)
set(PUBLIC_GPU_LIBS CUDA::cudart CUDA::cublasLt CUDA::nvml CUDA::cuFile cudnn_frontend nvidia::nccl ${NVML_LIBS})

message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
message(STATUS "cuBLAS Library: ${CUDA_cublas_LIBRARY}")
message(STATUS "cuBLASLt Library: ${CUDA_cublasLt_LIBRARY}")
