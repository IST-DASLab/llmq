// AMD compatibility header. When compiling for rocm, this header will get included into every translation unit
// Based on Anthonix's work for llm.c: https://github.com/anthonix/llm.c/blob/master/llmc/amd_common.cuh

#include <hip/hip_bf16.h>
#include <hip/hip_bfloat16.h>

static __device__ __forceinline__ float __ldcs(const float *addr) {
    return __builtin_nontemporal_load(addr);
}

static __device__ __forceinline__ int4 __ldcs(const int4 *addr) {
    const int *a = (const int *) addr;
    return make_int4(
        __builtin_nontemporal_load(a),
        __builtin_nontemporal_load(a+1),
        __builtin_nontemporal_load(a+2),
        __builtin_nontemporal_load(a+3)
        );
}

static __device__ __forceinline__ void __stcs(float *addr, float val) {
    __builtin_nontemporal_store(val, addr);
}

static __device__ __forceinline__ void __stcs(hip_bfloat16 *addr, hip_bfloat16 val) {
    __builtin_nontemporal_store(*reinterpret_cast<short*>(&val), reinterpret_cast<short*>(addr));
}

static __device__ __forceinline__ hip_bfloat16 __float2bfloat16_rn(float f) {
    return hip_bfloat16(f);
}

static __device__ __forceinline__ unsigned int atomicMax_block(unsigned int* addr, unsigned int value) {
    return atomicMax(addr, value);
}
