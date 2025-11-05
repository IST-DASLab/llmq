// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Based on llm.c https://github.com/karpathy/llm.c

/*
The GPT-2 Encoder, which combines two encodings: token and position
In the forward pass, both encodings are added together
In the backward pass, the gradients flow to both, handled by different kernels
*/
// From llm.c https://github.com/karpathy/llm.c
#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <vector>

#include "utilities/utils.h"
#include "utilities/vec.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

template<typename floatX>
__global__ void encoder_forward_kernel3(floatX* out,
                               const int* inp, const floatX* wte, const floatX* wpe,
                               int B, int T, int C) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    int N = B * T * C;
    if (idx >= N) { return; }

    int bt = idx / C;
    int b = bt / T;
    int t = bt % T;
    int c = idx % C;

    int ix = inp[b * T + t];

    floatX* out_btc = out + b * T * C + t * C + c;
    const floatX* wte_ix = wte + ix * C + c;
    const floatX* wpe_tc = wpe + t * C + c;

    x128 packed_out;
    x128 wte128 = load128cs(wte_ix);
    x128 wpe128 = load128cs(wpe_tc);
    for (int k = 0; k < x128::size; k++) {
        packed_out[k] = (floatX)((float)wte128[k] + (float)wpe128[k]);
    }
    store128(out_btc, packed_out);
}

// same kernel but without the positional encoder
template<typename floatX>
__global__ void encoder_forward_kernel3_nowpe(floatX* out,
                               const int* inp, const floatX* wte,
                               int B, int T, int C, int V) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    int N = B * T * C;
    if (idx >= N) { return; }
    int bt = idx / C;
    int b = bt / T;
    int t = bt % T;
    int c = idx % C;
    int ix = inp[b * T + t];
    assert(0 <= ix && ix < V);
    x128 wte128 = x128::load(wte + ix * C + c);
    wte128.store(out + b * T * C + t * C + c);
}

// ----------------------------------------------------------------------------
// kernel launchers
template<class floatX>
void encoder_forward_imp(floatX* out,
                         const int* inp, const floatX* wte, const floatX* wpe,
                         int B, int T, int C, int V, cudaStream_t stream) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    NVTX_RANGE_FN();
    constexpr int block_size = 256;
    const int N = B * T * C;
    const int grid_size = div_ceil(N, (int)(block_size * x128::size));
    if (wpe == nullptr) {
        // Llama 3 does not use positional encoder
        encoder_forward_kernel3_nowpe<<<grid_size, block_size, 0, stream>>>(out, inp, wte, B, T, C, V);
    } else {
        // GPT-2 does, so we use the full encoder kernel
        // encoder_forward_kernel3<<<grid_size, block_size, 0, stream>>>(out, inp, wte, wpe, B, T, C);
    }
    CUDA_CHECK(cudaGetLastError());
}

void encoder_forward(float* out, const int* inp, const float* wte, const float* wpe, int B, int T, int C, int V, cudaStream_t stream) {
    encoder_forward_imp(out, inp, wte, wpe, B, T, C, V, stream);
}

void encoder_forward(nv_bfloat16* out, const int* inp, const nv_bfloat16* wte, const nv_bfloat16* wpe, int B, int T, int C, int V, cudaStream_t stream) {
    encoder_forward_imp(out, inp, wte, wpe, B, T, C, V, stream);
}


template <typename floatX, int BLOCK_SIZE=256>
__global__ void wte_backward_kernel(floatX* dwte,
                                    const int4* bucket_info, const int* workload_indices, const floatX* dout, const int* inp,
                                    unsigned int seed, int B, int T, int C) {
    // In order to be deterministic, we preprocess the inputs on the cpu into "buckets"
    // Each bucket corresponds to (WARP_SIZE * x128::size) channels for a single vocabulary token
    // Each thread handles x128::size channels, e.g. 256 per warp for BF16
    // Each block handles (BLOCK_SIZE / WARP_SIZE) elements in a single bucket in parallel
    // If a bucket has less than 8 elements, some warps will return immediately
    // If a bucket has more than 8 elements, we will loop over all of them
    // The buckets are sorted on the CPU so the largest buckets start 1st
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    int bucket = blockIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int c_per_warp = 32 * x128::size;

    int bucket_start_idx = bucket_info[bucket].x;
    int bucket_size = bucket_info[bucket].y;
    int bucket_ix = bucket_info[bucket].z;
    int c = bucket_info[bucket].w * c_per_warp + (lane_id * x128::size);

    // Each thread handles "x128::size" channels, so at fp8, each warp would handle 512 channels
    // If C is not a multiple of this (e.g. 768), some buckets/c_groups cannot use the entire warp
    if (c >= C) { return; }
    // Exit early if this is a small bucket and this warp doesn't have any items to process
    if (warp_id >= bucket_size) { return; }

    float accum[x128::size] = {0.0f};
    __shared__ float accum_shared[x128::size * BLOCK_SIZE];

    for(int item = warp_id; item < bucket_size; item += BLOCK_SIZE/32) {
        int bt = workload_indices[bucket_start_idx + item];

        const floatX* dout_btc = dout + bt * C + c;
        x128 packed_inp1 = x128::load_cs(dout_btc);
        for (int k = 0; k < packed_inp1.size; k++) {
            accum[k] += (float)packed_inp1[k];
        }
    }

    if (warp_id != 0) {
        // we accumulate into warp 0, so only the other warps need to write to shared memory
        for (int k = 0; k < x128::size; k++) {
            accum_shared[threadIdx.x + k * BLOCK_SIZE] = accum[k];
        }
        return; // only warp 0 is needed after writing to shared memory
    }

    // Read dwte for warp 0 even if other warps are not finished yet to maximise latency tolerance
    floatX* dwte_ix = dwte + bucket_ix * C + c;
    x128 packed_in_out = x128::load(dwte_ix);

    // note: threads which have returned are considered synchronised by CUDA so no risk of deadlock
    __syncthreads();

    // Accumulate into warp 0's registers by reading the values of the other warps in shared memory
    for (int i = threadIdx.x+32; i < min(BLOCK_SIZE, bucket_size*32); i += 32) {
        for (int k = 0; k < x128::size; k++) {
            accum[k] += accum_shared[i + k * BLOCK_SIZE];
        }
    }

    // Add the result to dwte and write back to global memory (read-modify-write)
    for (unsigned int k = 0; k < x128::size; k++) {
        // We use stochastic rounding to go from FP32 to BF16
        // The seed is deterministic and unique for each parameter to guarantee we have determinism AND
        // to avoid **potential** issues with positionX int SquirrelNoise5 argument overflowing which is UB
        // and that somehow messing the quality of random numbers
        // TODO  re-enable  this
        // stochastic_rounding(accum[k] + (float)packed_in_out[k], &packed_in_out[k], seed + bucket * 32 + threadIdx.x + k);
        packed_in_out[k] = accum[k] + (float)packed_in_out[k];
    }
    packed_in_out.store(dwte_ix);
}

// Fully deterministic (see comments in wte_backward_kernel and wpe_backward_kernel for more details)
template<class floatX>
void encoder_backward_imp(floatX* dwte, int* scratch, // gpu outputs & scratch
                      int* workload_indices, int4* bucket_info,    // cpu scratch buffers
                      const floatX* dout, const int* inp, const int* inputs_cpu, // cpu/gpu inputs
                      int B, int T, int C, unsigned int seed, cudaStream_t stream) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    NVTX_RANGE_FN();

    int num_c_groups = div_ceil((size_t)C, x128::size * 32);
    assert(B*T*num_c_groups * (sizeof(int4)+sizeof(int)) <= B*T*3*C * sizeof(floatX));

    // Step 1: Sort inputs into buckets
    int total_items = 0;
    std::unordered_map<uint64_t, std::vector<uint64_t>> buckets;
    for (uint64_t bt = 0; bt < B * T; bt++) {
        for (uint64_t c_group = 0; c_group < num_c_groups; c_group++) {
            // todo - passing c_group/inputs_cpu[bt] in data to avoid a second hash lookup is a bit hacky
            uint64_t data = bt + (c_group<<32ULL) + ((uint64_t)inputs_cpu[bt]<<42ULL);
            buckets[c_group + num_c_groups * inputs_cpu[bt]].push_back(data);
            total_items++;
        }
    }

    // Step 2: Sort buckets by size in descending order
    // this is so the largest buckets are processed first by the GPU
    // otherwise, if they started late, they would still be running with the rest of the GPU idle
    std::vector<std::pair<uint64_t, std::vector<uint64_t>>> sortedBuckets(buckets.begin(), buckets.end());
    std::sort(sortedBuckets.begin(), sortedBuckets.end(), // ugly because we don't have a typedef for the std::pair
              [](const std::pair<uint64_t, std::vector<uint64_t>>& a, const std::pair<uint64_t, std::vector<uint64_t>>& b) {
                  return a.second.size() > b.second.size();
              });

    int num_buckets = buckets.size();
    int bucket_index = 0;
    int workload_index = 0;
    for (const auto& bucket : sortedBuckets) {
        bucket_info[bucket_index].x = workload_index; // bucket start
        bucket_info[bucket_index].y = bucket.second.size(); // bucket size
        bucket_info[bucket_index].z = (bucket.second[0] >> 42ULL) & ((1ULL<<20ULL)-1); // bucket ix
        bucket_info[bucket_index].w = (bucket.second[0] >> 32ULL) & ((1ULL<<10ULL)-1); // bucket c

        for (uint64_t idx : bucket.second) {
            workload_indices[workload_index++] = (int)(idx & ((1ULL<<31ULL)-1ULL));
        }
        bucket_index++;
    }

    // Step 3: Copy data from host to device (async until the last one to avoid synchronising CPU/GPU twice)
    // todo - could use CUDA events (even without streams) to avoid CPU/GPU synchronisation completely
    int4* d_bucket_info = (int4*)scratch;
    int*  d_workload_indices = (int*)(scratch + B*T*num_c_groups * 4);
    CUDA_CHECK(cudaMemcpyAsync(d_bucket_info, bucket_info, num_buckets * sizeof(int4), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_workload_indices, workload_indices, total_items * sizeof(int), cudaMemcpyHostToDevice, stream));

    // Launch wte kernel
    // todo - profile block sizes on more content (depends on number of buckets and on GPU?)
    wte_backward_kernel<floatX, 256><<<num_buckets, 256, 0, stream>>>(dwte, d_bucket_info, d_workload_indices, dout, inp, seed, B, T, C);
    CUDA_CHECK(cudaGetLastError());
}

void encoder_backward(float* dwte, int* scratch, // gpu outputs & scratch
                      int* workload_indices, int4* bucket_info,    // cpu scratch buffers
                      const float* dout, const int* inp, const int* inputs_cpu, // cpu/gpu inputs
                      int B, int T, int C, unsigned int seed, cudaStream_t stream) {
    encoder_backward_imp(dwte, scratch, workload_indices, bucket_info, dout, inp, inputs_cpu, B, T, C, seed, stream);
}

void encoder_backward(nv_bfloat16* dwte, int* scratch, // gpu outputs & scratch
                      int* workload_indices, int4* bucket_info,    // cpu scratch buffers
                      const nv_bfloat16* dout, const int* inp, const int* inputs_cpu, // cpu/gpu inputs
                      int B, int T, int C, unsigned int seed, cudaStream_t stream) {
    encoder_backward_imp(dwte, scratch, workload_indices, bucket_info, dout, inp, inputs_cpu, B, T, C, seed, stream);
}
