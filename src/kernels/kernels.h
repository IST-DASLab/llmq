// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_KERNELS_KERNELS_H
#define LLMQ_SRC_KERNELS_KERNELS_H

#include <cstdint>
#include <optional>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

struct cudaDeviceProp;
typedef struct cudnnContext* cudnnHandle_t;
typedef struct cublasLtContext* cublasLtHandle_t;

struct Tensor;
enum class ETensorDType: int;

enum class EMMTranspose { TT, TN, NT, NN };

void encoder_forward(float* out, const int* inp, const float* wte, const float* wpe, int B, int T, int C, int V, cudaStream_t stream);
void encoder_forward(nv_bfloat16* out, const int* inp, const nv_bfloat16* wte, const nv_bfloat16* wpe, int B, int T, int C, int V, cudaStream_t stream);
void encoder_forward(Tensor& out, const Tensor& inp, const Tensor& wte, const Tensor& wpe, int B, int T, int C, int V, cudaStream_t stream);

void encoder_backward(float* dwte, int* scratch,
                      int* workload_indices, int4* bucket_info,
                      const float* dout, const int* inp, const int* inputs_cpu,
                      int B, int T, int C, unsigned int seed, cudaStream_t stream, cudaEvent_t sync_event, cudaStream_t copy_stream);
void encoder_backward(nv_bfloat16* dwte, int* scratch,
                      int* workload_indices, int4* bucket_info,
                      const nv_bfloat16* dout, const int* inp, const int* inputs_cpu,
                      int B, int T, int C, unsigned int seed, cudaStream_t stream, cudaEvent_t sync_event, cudaStream_t copy_stream);

// The kernel runs on `stream`, but the bucket info that gets generated on CPU to enable efficient determinism
// can be copied using `copy_stream`, so the kernel launch does not have to wait.
void encoder_backward(Tensor& dwte, Tensor& scratch,
                      Tensor& workload_indices, Tensor& bucket_info,
                      const Tensor& dout, const Tensor& inp, const Tensor& inputs_cpu,
                      int B, int T, int C, unsigned int seed, cudaStream_t stream, cudaEvent_t sync_event, cudaStream_t copy_stream);

void rmsnorm_forward(float* out, float* rms, const float* inp, const float* weight, float* abs_max_ptr, float epsilon, int B, int T, int C, cudaStream_t stream);
void rmsnorm_forward(nv_bfloat16* out, float* rms, const nv_bfloat16* inp, const nv_bfloat16* weight, float* abs_max_ptr, float epsilon, int B, int T, int C, cudaStream_t stream);
void rmsnorm_forward(Tensor& out, Tensor& rms, const Tensor& inp, const Tensor& weight, float* abs_max_ptr, float epsilon, int B, int T, int C, cudaStream_t stream);

int get_rmsnorm_backward_scratch_size(int C, const cudaDeviceProp& dp);
void rmsnorm_backward(float* dinp, float* dweight, std::byte* scratch, const float* dresidual, const float* dout, const float* inp, const float* weight, const float* rstd, float* abs_max_ptr,
                      int B, int T, int C, const cudaDeviceProp& dp, cudaStream_t stream);
void rmsnorm_backward(nv_bfloat16* dinp, nv_bfloat16* dweight, std::byte* scratch, const nv_bfloat16* dresidual, const nv_bfloat16* dout, const nv_bfloat16* inp, const nv_bfloat16* weight, const float* rstd, float* abs_max_ptr,
                      int B, int T, int C, const cudaDeviceProp& dp, cudaStream_t stream);
void rmsnorm_backward(Tensor& dinp, Tensor& dweight, Tensor& scratch, const Tensor& dresidual, const Tensor& dout, const Tensor& inp, const Tensor& weight, const Tensor& rstd, float* abs_max_ptr,
                      int B, int T, int C,  const cudaDeviceProp& dp, cudaStream_t stream);

void fused_residual_rmsnorm_forward(float* residual, float* normed, float* rrms, const float* inp1, const float* inp2, const float* weight, float* abs_max_ptr,
                                    float epsilon, int N, int C, cudaStream_t stream);
void fused_residual_rmsnorm_forward(nv_bfloat16* residual, nv_bfloat16* normed, float* rrms, const nv_bfloat16* inp1, const nv_bfloat16* inp2, const nv_bfloat16* weight, float* abs_max_ptr,
                                    float epsilon, int N, int C, cudaStream_t stream);
void fused_residual_rmsnorm_forward(Tensor& residual, Tensor& normed, Tensor& rrms, const Tensor& inp1, const Tensor& inp2, const Tensor& weight, float* abs_max_ptr,
                                    float epsilon, int N, int C, cudaStream_t stream);

void matmul(float* c, const float* a, const float* b, const float* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream);

void matmul(float* c, const nv_bfloat16* a, const nv_bfloat16* b, const float* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream);

void matmul(float* c, const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, const float* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream);

void matmul(float* c, const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream);

void matmul(float* c, const __nv_fp8_e4m3* a, const __nv_fp8_e5m2* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream);

void matmul(nv_bfloat16* c, const nv_bfloat16* a, const nv_bfloat16* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream);

void matmul(nv_bfloat16* c, const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream);

void matmul(nv_bfloat16* c, const __nv_fp8_e4m3* a, const __nv_fp8_e5m2* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream);

void matmul(Tensor& c, const Tensor& a, const Tensor& b, const Tensor& bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, Tensor& workspace,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream);

void add_bias(float* out, const float* bias, int B, int T, int OC, cudaStream_t stream);
void add_bias(nv_bfloat16* out, const nv_bfloat16* bias, int B, int T, int OC, cudaStream_t stream);
int get_bias_backward_scratch_size(ETensorDType dtype, int OC, const cudaDeviceProp& dp);
void backward_bias(float* dbias, const float* dout, const float* scale_a, const float* scale_b, float* dbias_buffer, int B, int T, int OC, const cudaDeviceProp& dp, cudaStream_t stream);
void backward_bias(nv_bfloat16* dbias, const nv_bfloat16* dout, const float* scale_a, const float* scale_b, float* dbias_buffer, int B, int T, int OC, const cudaDeviceProp& dp, cudaStream_t stream);
void backward_bias(nv_bfloat16* dbias, const __nv_fp8_e4m3* dout, const float* scale_a, const float* scale_b, float* dbias_buffer, int B, int T, int OC, const cudaDeviceProp& dp, cudaStream_t stream);
void backward_bias(nv_bfloat16* dbias, const __nv_fp8_e5m2* dout, const float* scale_a, const float* scale_b, float* dbias_buffer, int B, int T, int OC, const cudaDeviceProp& dp, cudaStream_t stream);
void backward_bias(Tensor& dbias, const Tensor& dout, const float* scale_a, const float* scale_b, Tensor& dbias_buffer, int B, int T, int OC, const cudaDeviceProp& dp, cudaStream_t stream);


void precompute_freqs_cis(float *freqs_cis, int dim, int end, float theta);
void precompute_freqs_cis(half *freqs_cis, int dim, int end, float theta);
void rope_forward(float* out, const float* in, const float *freqs_cis, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream);
void rope_forward(nv_bfloat16* out, const nv_bfloat16* in, const half *freqs_cis, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream);
void rope_forward(Tensor& out, const Tensor& in, const Tensor& freqs_cis, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream);
void rope_backward(float* dinp, const float* dout, const float *freqs_cis, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream);
void rope_backward(nv_bfloat16* dinp, const nv_bfloat16* dout, const half *freqs_cis, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream);
void rope_backward(Tensor& dinp, const Tensor& dout, const Tensor& freqs_cis, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream);

// swiglu assumes that input is the concatenation of gate and up projection.
void swiglu_forward(nv_bfloat16* out, const nv_bfloat16* inp, float* abs_max_ptr, int B, int T, int C, cudaStream_t stream);
void swiglu_forward(float* out, const float* inp, float* abs_max_ptr, int B, int T, int C, cudaStream_t stream);
void swiglu_forward(Tensor& out, const Tensor& inp, float* abs_max_ptr, int B, int T, int C, cudaStream_t stream);

void swiglu_forward_quant(__nv_fp8_e4m3* out, float* scale_ptr,const nv_bfloat16* inp, const float* abs_max_ptr, int B, int T, int C, cudaStream_t stream);
void swiglu_forward_quant(Tensor& out, float* scale_ptr, const Tensor& inp, const float* abs_max_ptr, int B, int T, int C, cudaStream_t stream);

void swiglu_backward(nv_bfloat16* dinp, const nv_bfloat16* dout, const nv_bfloat16* inp, float* abs_max, int B, int T, int C, cudaStream_t stream);
void swiglu_backward(float* dinp, const float* dout, const float* inp, float* abs_max, int B, int T, int C, cudaStream_t stream);
void swiglu_backward(Tensor& dinp, const Tensor& dout, const Tensor& inp, float* abs_max, int B, int T, int C, cudaStream_t stream);

void attention_forward_cudnn(nv_bfloat16* out,  // output: (B, T, Nq, HS)
                             float* stats, // output for backward pass: (B, Hq, T)
                             const nv_bfloat16* inp,  // input: (B, T, Hq + 2Hkv, HS) QKV
                             std::byte* workspace, cudnnHandle_t handle,
                             int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream);

void attention_forward_cudnn(float* out,  // output: (B, T, Nq, HS)
                             float* stats, // output for backward pass: (B, Hq, T)
                             const float* inp,  // input: (B, T, Hq + 2Hkv, HS) QKV
                             std::byte* workspace, cudnnHandle_t handle,
                             int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream);

void attention_forward_cudnn(Tensor& out,  // output: (B, T, Nq, HS)
                             Tensor& stats, // output for backward pass: (B, Hq, T)
                             const Tensor& inp,  // input: (B, T, Hq + 2Hkv, HS) QKV
                             Tensor& workspace, cudnnHandle_t handle,
                             int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream);

std::size_t cudnn_get_workspace_size(int B, int T, int Hq, int Hkv, int HS, cudnnHandle_t handle);
void attention_backward_cudnn(nv_bfloat16* dqkv, const float* stats,
                              const nv_bfloat16* out, const nv_bfloat16* dout, const nv_bfloat16* qkv,
                              std::byte* workspace, cudnnHandle_t handle,
                              int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream);
void attention_backward_cudnn(Tensor& dqkv, const Tensor& stats,
                              const Tensor& out, const Tensor& dout, const Tensor& qkv,
                              Tensor& workspace, cudnnHandle_t handle,
                              int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream);

void fused_classifier(float* logits, float* losses, float* lse,
                      float dloss, const int* targets, float z_reg,
                      int BT, int V, int P, bool write_dlogits, cudaStream_t stream);
void fused_classifier(nv_bfloat16* logits, float* losses, float* lse,
                      float dloss, const int* targets, float z_reg,
                      int BT, int V, int P, bool write_dlogits, cudaStream_t stream);
void fused_classifier(Tensor& logits, Tensor& losses, Tensor& lse,
                      float dloss, const Tensor& targets, float z_reg,
                      int BT, int V, int P, bool write_dlogits, cudaStream_t stream);

void reduce_lse_stats(float* result, const float* in, long N, bool first_step, cudaStream_t stream);

int get_max_num_block_sums(const cudaDeviceProp& dp);
void global_norm_squared(float* out, const float* values, size_t count, const cudaDeviceProp& dp, cudaStream_t stream);
void global_norm_squared(float* out, const nv_bfloat16* values, size_t count, const cudaDeviceProp& dp, cudaStream_t stream);
void global_norm_squared(Tensor& out, const Tensor& values, size_t count, const cudaDeviceProp& dp, cudaStream_t stream);
/// puts norm squared in out[0], norm in out_cpu, and grad scale factor in out[1]
void global_norm_sqrt(float* out, float* out_cpu, float grad_clip, const cudaDeviceProp& dp, cudaStream_t stream);

void deterministic_sum(float* out, const float* values, std::size_t count, cudaStream_t stream);
void deterministic_sum(float* out, const nv_bfloat16* values, std::size_t count, cudaStream_t stream);


void adamw_update(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                  float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  const float* grad_scale, float* abs_max, unsigned int seed, cudaStream_t stream);

void adamw_update(nv_bfloat16* params_memory, const nv_bfloat16* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                  float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  const float* grad_scale, float* abs_max, unsigned int seed, cudaStream_t stream);

void adamw_update(nv_bfloat16* params_memory, const nv_bfloat16* grads_memory, nv_bfloat16* m_memory, float* v_memory, size_t num_parameters,
                  float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  const float* grad_scale, float* abs_max, unsigned int seed, cudaStream_t stream);

void adamw_update(nv_bfloat16* params_memory, const nv_bfloat16* grads_memory, nv_bfloat16* m_memory, nv_bfloat16* v_memory, size_t num_parameters,
                  float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  const float* grad_scale, float* abs_max, unsigned int seed, cudaStream_t stream);

void adamw_update(nv_bfloat16* params_memory, const nv_bfloat16* grads_memory, __nv_fp8_e4m3* m_memory, nv_bfloat16* v_memory, size_t num_parameters,
                  float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  const float* grad_scale, float* m_scales, float* abs_max, unsigned int seed, cudaStream_t stream);

void adamw_update(Tensor& params_memory, const Tensor& grads_memory, Tensor& m_memory, Tensor& v_memory, size_t num_parameters,
                  float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  const float* grad_scale, Tensor& m_scales, float* abs_max, unsigned int seed, cudaStream_t stream);

// quantization
void abs_max(float* scale, const float* in, long N, const cudaDeviceProp& dp, cudaStream_t stream);
void abs_max(float* scale, const nv_bfloat16* in, long N, const cudaDeviceProp& dp, cudaStream_t stream);
void abs_max(float* scale, const Tensor& in, long N, const cudaDeviceProp& dp, cudaStream_t stream);

void quantize_with_abs_max(nv_bfloat16* out, float* scale_ptr, const float* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_with_abs_max(std::int8_t* out, float* scale_ptr, const float* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_with_abs_max(__nv_fp8_e4m3* out, float* scale_ptr, const float* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_with_abs_max(__nv_fp8_e5m2* out, float* scale_ptr, const float* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_with_abs_max(std::int8_t* out, float* scale_ptr, const nv_bfloat16* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_with_abs_max(__nv_fp8_e4m3* out, float* scale_ptr, const nv_bfloat16* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_with_abs_max(__nv_fp8_e5m2* out, float* scale_ptr, const nv_bfloat16* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_with_abs_max(Tensor& out, float* scale_ptr, const Tensor& in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream);

void quantize_and_transpose_with_abs_max(nv_bfloat16* out, float* scale_ptr, const float* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_and_transpose_with_abs_max(std::int8_t* out, float* scale_ptr, const float* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_and_transpose_with_abs_max(__nv_fp8_e4m3* out, float* scale_ptr, const float* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_and_transpose_with_abs_max(std::int8_t* out, float* scale_ptr, const nv_bfloat16* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_and_transpose_with_abs_max(__nv_fp8_e4m3* out, float* scale_ptr, const nv_bfloat16* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_and_transpose_with_abs_max(Tensor& out, float* scale_ptr, const Tensor& in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream);

void transpose(float* dst, const float* src, int rows, int cols, cudaStream_t stream);
void transpose(__nv_fp8_e4m3* dst, const __nv_fp8_e4m3* src, int rows, int cols, cudaStream_t stream);
void transpose(__nv_fp8_e5m2* dst, const __nv_fp8_e5m2* src, int rows, int cols, cudaStream_t stream);
void transpose(nv_bfloat16* dst, const nv_bfloat16* src, int rows, int cols, cudaStream_t stream);
void transpose(Tensor& dst, const Tensor& src, int rows, int cols, cudaStream_t stream);

void vector_add_sr(float* dest, const float* left, const float* right, float scale, long nelem, unsigned seed, cudaStream_t stream);
void vector_add_sr(nv_bfloat16* dest, const nv_bfloat16* left, const nv_bfloat16* right, float scale, long nelem, unsigned seed, cudaStream_t stream);
void vector_add_sr(Tensor& dest, const Tensor& left, const Tensor& right, float scale, long nelem, unsigned seed, cudaStream_t stream);

//! \fn void vector_reduce_sr(Tensor& dest, const Tensor& src, float scale, int n_shards, int skip, long nelem, bool accumulate, unsigned seed, cudaStream_t stream);
//! \brief Reduce shards of tensor using stochastic rounding
//! \details Interprets `src` as a tensor of `n_shard` shards of size `nelem` each. The shards are summed together, and the result is either written to (`accumulate = false`)
//! or added into (`accumulate = true`) `dest`, after being scaled by `scale`. All intermediate calculations are done in float precision, and stochastic rounding using the
//! provided `seed` is applied before writing to `dest`. The `skip` parameter allows to skip one of the shards. Set to -1 to disable skipping.
void vector_reduce_sr(float* dest, const float* src, float scale, int n_shards, int skip, long nelem, bool accumulate, unsigned seed, cudaStream_t stream);
void vector_reduce_sr(nv_bfloat16* dest, const nv_bfloat16* src, float scale, int n_shards, int skip, long nelem, bool accumulate, unsigned seed, cudaStream_t stream);
void vector_reduce_sr(Tensor& dest, const Tensor& src, float scale, int n_shards, int skip, long nelem, bool accumulate, unsigned seed, cudaStream_t stream);

void fill_normal(float* dst, std::size_t count, float mean, float std, unsigned long long seed, unsigned long long subsequence, cudaStream_t stream);
void fill_normal(nv_bfloat16* dst, std::size_t count, float mean, float std, unsigned long long seed, unsigned long long subsequence, cudaStream_t stream);
void fill_normal(Tensor& dest, std::size_t count, float mean, float std, unsigned long long seed, unsigned long long subsequence, cudaStream_t stream);

void fill_constant(float* dst, float value, std::size_t count, cudaStream_t stream);
void fill_constant(nv_bfloat16* dst, nv_bfloat16 value, std::size_t count, cudaStream_t stream);
void fill_constant(Tensor& dest, float value, std::size_t count, cudaStream_t stream);

void convert_dtype(float* target, const nv_bfloat16* source, std::size_t size);
void convert_dtype(nv_bfloat16* target, const float* source, std::size_t size);
void convert_dtype(nv_bfloat16* target, const half* source, std::size_t size);

#endif //LLMQ_SRC_KERNELS_KERNELS_H
