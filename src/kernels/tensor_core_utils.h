#include "utilities/dtype.h"
#include <vector_types.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline_primitives.h>

template<typename T>
struct m16_n16_k32_a_fragment {
    uint4 v;
};

template<typename T>
struct m16_n16_k32_b_fragment {
    uint4 v;
};

template<typename AccDType>
struct m16_n16_k32_c_fragment {
    AccDType v[8] = {0.f, 0.f, 0.f, 0.f};
};

template<typename>
constexpr char ptx_type_name[] = "unknown_dtype";

template<>
constexpr char ptx_type_name<float>[4] = "f32";

template<>
constexpr char ptx_type_name<half>[4] = "f16";

template<>
constexpr char ptx_type_name<nv_bfloat16>[5] = "bf16";

template<>
constexpr char ptx_type_name<__nv_fp8_e4m3>[5] = "e4m3";

template<>
constexpr char ptx_type_name<__nv_fp8_e5m2>[5] = "e5m2";


__device__ __forceinline__ void global_to_shared_16_32_swizzle(uint4** shared, const uint4** global, int stride) {
    int col = threadIdx.x % 2;
    int row = threadIdx.x / 2;

    int g8 = threadIdx.x / 8;
    int t8 = threadIdx.x % 8;

    *shared = *shared + (t8 ^ g8) + 8 * g8;
    *global = *global + row * stride + col;
}

__device__ __forceinline__ int load_address(int row, int col) {
    int lin = col + 2 * row;
    int g8 = lin / 8;
    int t8 = lin % 8;
    return (t8 ^ g8) + 8 * g8;
}

__device__ __forceinline__ int2 ldmatrix_offsets() {
    int t8 = threadIdx.x % 8;
    int g8 = threadIdx.x / 8;
    int a = load_address(t8 + 8 * (g8%2), g8 / 2);
    int b = load_address(t8 + 8 * (g8/2), g8 % 2);
    return make_int2(a, b);
}

__device__ __forceinline__ void ptx_ldmatrix(uint4& dst, const void* src) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w)
        : "l"(__cvta_generic_to_shared(src))
        );
}

template<typename AccDType>
__device__ __forceinline__ void store_fragment_row_major_sync(m16_n16_k32_c_fragment<AccDType>& c, nv_bfloat16* ptr, int row_stride) {
    int g4 = threadIdx.x / 4;
    int c4 = threadIdx.x % 4;
    nv_bfloat162* vptr = reinterpret_cast<nv_bfloat162*>(ptr);
    row_stride /= 2;
    vptr[row_stride * (g4 + 0) + c4 + 0] = make_bfloat162((nv_bfloat16)c.v[0], (nv_bfloat16)c.v[1]);
    vptr[row_stride * (g4 + 8) + c4 + 0] = make_bfloat162((nv_bfloat16)c.v[2], (nv_bfloat16)c.v[3]);

    vptr[row_stride * (g4 + 0) + c4 + 4] = make_bfloat162((nv_bfloat16)c.v[4], (nv_bfloat16)c.v[5]);
    vptr[row_stride * (g4 + 8) + c4 + 4] = make_bfloat162((nv_bfloat16)c.v[6], (nv_bfloat16)c.v[7]);
}

template<typename AType, typename BType>
__device__ __forceinline__ void mma_m16_n16_k32_sync(m16_n16_k32_c_fragment<float>& d,
                                                     m16_n16_k32_a_fragment<AType> a,
                                                     m16_n16_k32_b_fragment<BType> b,
                                                     m16_n16_k32_c_fragment<float> c) {
    static_assert(sizeof(AType) == sizeof(BType), "a and b type must have the same size");

    constexpr int k = 32 / sizeof(AType);
    asm volatile("mma.sync.aligned.m16n8k%26.row.col.f32.%24.%25.f32 "
                 "{%0, %1, %2, %3},"
                 "{%8, %9, %10, %11},"
                 "{%12, %13},"
                 "{%16, %17, %18, %19};\n"
                 "mma.sync.aligned.m16n8k%26.row.col.f32.%24.%25.f32 "
                 "{%4, %5, %6, %7},"
                 "{%8, %9, %10, %11},"
                 "{%14, %15},"
                 "{%20, %21, %22, %23};\n"
        : "=f"(d.v[0]), "=f"(d.v[1]), "=f"(d.v[2]), "=f"(d.v[3]),
          "=f"(d.v[4]), "=f"(d.v[5]), "=f"(d.v[6]), "=f"(d.v[7])
        : "r"(a.v.x), "r"(a.v.y), "r"(a.v.z), "r"(a.v.w),
          "r"(b.v.x), "r"(b.v.y), "r"(b.v.z), "r"(b.v.w),
          "f"(c.v[0]), "f"(c.v[1]), "f"(c.v[2]), "f"(c.v[3]),
          "f"(c.v[4]), "f"(c.v[5]), "f"(c.v[6]), "f"(c.v[7]),
          "C"(ptx_type_name<AType>), "C"(ptx_type_name<BType>), "n"(k));
}

template<typename AType, typename BType>
__device__ __forceinline__ void mma_m16_n16_k32_sync(m16_n16_k32_c_fragment<half>& d,
                                                     m16_n16_k32_a_fragment<AType> a,
                                                     m16_n16_k32_b_fragment<BType> b,
                                                     m16_n16_k32_c_fragment<half> c) {
    auto to_raw = [](half& h) -> unsigned int& {
        return *reinterpret_cast<unsigned int*>(&h);
    };
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.%10.%11.f16 "
                 "{%0, %1},"
                 "{%2, %3, %4, %5},"
                 "{%6, %7},"
                 "{%8, %9};"
        : "=r"(to_raw(d.v[0])), "=r"(to_raw(d.v[2]))
        : "r"(a.v.x), "r"(a.v.y), "r"(a.v.z), "r"(a.v.w),
          "r"(b.v.x), "r"(b.v.y),
          "r"(to_raw(c.v[0])), "r"(to_raw(c.v[2])),
          "C"(ptx_type_name<AType>), "C"(ptx_type_name<BType>));

    asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.%10.%11.f16 "
                 "{%0, %1},"
                 "{%2, %3, %4, %5},"
                 "{%6, %7},"
                 "{%8, %9};"
        : "=r"(to_raw(d.v[4])), "=r"(to_raw(d.v[6]))
        : "r"(a.v.x), "r"(a.v.y), "r"(a.v.z), "r"(a.v.w),
          "r"(b.v.z), "r"(b.v.w),
          "r"(to_raw(c.v[4])), "r"(to_raw(c.v[6])),
          "C"(ptx_type_name<AType>), "C"(ptx_type_name<BType>));
}
