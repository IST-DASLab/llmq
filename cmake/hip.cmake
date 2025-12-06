set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")

list(APPEND CMAKE_PREFIX_PATH
        ${ROCM_PATH}
        ${ROCM_PATH}/lib/cmake
        ${ROCM_PATH}/lib/cmake/hip
        ${ROCM_PATH}/lib/cmake/hipblas
        ${ROCM_PATH}/lib/cmake/rocblas
)

find_package(hip REQUIRED CONFIG PATHS ${ROCM_PATH})
find_package(rocblas REQUIRED CONFIG PATHS ${ROCM_PATH})
find_package(hipblas REQUIRED CONFIG PATHS ${ROCM_PATH})
find_package(MIOpen REQUIRED CONFIG PATHS ${ROCM_PATH})
find_package(RCCL REQUIRED CONFIG PATHS "${ROCM_PATH}")

find_program(HIPIFY_PERL hipify-perl PATHS ${ROCM_PATH}/bin REQUIRED)

# Hipifies a list of files, returns hipified file list
function(hipify_files OUT_VAR HIPIFY_OUTPUT_DIR SOURCE_DIR)
    set(HIPIFIED_FILES "")
    foreach(SOURCE_FILE ${ARGN})
        get_filename_component(FILE_EXT ${SOURCE_FILE} EXT)
        get_filename_component(ABS_SOURCE_FILE ${SOURCE_FILE} ABSOLUTE)
        file(RELATIVE_PATH REL_PATH ${SOURCE_DIR} ${ABS_SOURCE_FILE})
        get_filename_component(FILE_NAME ${SOURCE_FILE} NAME_WE)
        get_filename_component(FILE_EXT_ONLY ${SOURCE_FILE} EXT)
        get_filename_component(REL_DIR ${REL_PATH} DIRECTORY)

        set(OUTPUT_DIR ${HIPIFY_OUTPUT_DIR}/${REL_DIR})
        file(MAKE_DIRECTORY ${OUTPUT_DIR})

        # .cu -> .hip, others keep extension
        if(FILE_EXT STREQUAL ".cu")
            set(OUTPUT_FILE ${OUTPUT_DIR}/${FILE_NAME}.hip)
        else()
            set(OUTPUT_FILE ${OUTPUT_DIR}/${FILE_NAME}${FILE_EXT_ONLY})
        endif()

        add_custom_command(
                OUTPUT ${OUTPUT_FILE}
                COMMAND ${HIPIFY_PERL} ${ABS_SOURCE_FILE} > ${OUTPUT_FILE}.tmp
                COMMAND sed -e 's|<cuda_bf16\\.h>|<hip/hip_bf16.h>|g' -e 's|<cuda_fp16\\.h>|<hip/hip_fp16.h>|g' -e 's|<cuda_runtime\\.h>|<hip/hip_runtime.h>|g' ${OUTPUT_FILE}.tmp > ${OUTPUT_FILE}
                COMMAND ${CMAKE_COMMAND} -E rm ${OUTPUT_FILE}.tmp
                DEPENDS ${ABS_SOURCE_FILE}
                COMMENT "Hipifying ${REL_PATH} to ${OUTPUT_FILE}"
        )

        list(APPEND HIPIFIED_FILES ${OUTPUT_FILE})
    endforeach()

    set(${OUT_VAR} ${HIPIFIED_FILES} PARENT_SCOPE)
endfunction()

# Hipifies and entire target
function(hipify_target TARGET_NAME HIPIFY_OUTPUT_DIR)
    get_target_property(TARGET_SOURCES ${TARGET_NAME} SOURCES)
    get_target_property(TARGET_SOURCE_DIR ${TARGET_NAME} SOURCE_DIR)

    # update source files
    hipify_files(HIPIFIED_SOURCES ${HIPIFY_OUTPUT_DIR} ${TARGET_SOURCE_DIR} ${TARGET_SOURCES})
    set_target_properties(${TARGET_NAME} PROPERTIES SOURCES "${HIPIFIED_SOURCES}")

    # OK, now  we need to handle includes
    get_target_property(INCLUDE_DIRS ${TARGET_NAME} INCLUDE_DIRECTORIES)
    set(HIPIFIED_INCLUDE_DIRS "")
    foreach(INC_DIR ${INCLUDE_DIRS})
        get_filename_component(ABS_INC_DIR ${INC_DIR} ABSOLUTE)
        file(RELATIVE_PATH REL_INC ${TARGET_SOURCE_DIR} ${ABS_INC_DIR})
        # Check if relative path doesn't start with ../ (inside source tree)
        if(NOT REL_INC MATCHES "^\\.\\.")
            file(GLOB_RECURSE INC_FILES "${INC_DIR}/*.h" "${INC_DIR}/*.hpp" "${INC_DIR}/*.cuh")
            hipify_files(HIPIFIED_HEADERS ${HIPIFY_OUTPUT_DIR} ${TARGET_SOURCE_DIR} ${INC_FILES})
            target_sources(${TARGET_NAME} PRIVATE ${HIPIFIED_HEADERS})
            list(APPEND HIPIFIED_INCLUDE_DIRS ${HIPIFY_OUTPUT_DIR}/${REL_INC})
        else()
            list(APPEND HIPIFIED_INCLUDE_DIRS ${INC_DIR})
        endif()
    endforeach()

    set_target_properties(${TARGET_NAME} PROPERTIES INCLUDE_DIRECTORIES "${HIPIFIED_INCLUDE_DIRS}")
    # not really correct... but let's set up the hipified includes also for interface
    set_target_properties(${TARGET_NAME} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${HIPIFIED_INCLUDE_DIRS}")

    # enforce presence of compatibility header
    target_compile_options(${TARGET_NAME} PUBLIC -include "${CMAKE_SOURCE_DIR}/src/utilities/amd.h")
    message(STATUS "DIRS FOR ${TARGET_NAME}")
    message(STATUS "INC_DIR: ${HIPIFIED_INCLUDE_DIRS}")
endfunction()

set(PRIVATE_GPU_LIBS hip::host hip::device MIOpen roc::hipblas rccl)
set(PUBLIC_GPU_LIBS hip::host rccl)
