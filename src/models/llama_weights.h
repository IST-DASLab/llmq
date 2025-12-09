// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_LLAMA_WEIGHTS_H
#define LLMQ_LLAMA_WEIGHTS_H

#include <optional>
#include <vector>

#include "utilities/tensor.h"
#include "utilities/tensor_container.h"
#include "llama_config.h"

struct LLamaOptions;
struct LLamaRunState;
class TensorAllocator;
class NCCLCommunicator;
class DeviceMemoryStack;
enum class EAllocationType : int;
typedef struct CUevent_st* cudaEvent_t;

template<class TTensor>
struct sLLamaBlockWeights {
    using OTensor = std::optional<TTensor>;
    TTensor LN1_w;           // C
    TTensor LN2_w;           // C
    TTensor Attn_QKV_w;      // ((Hq + 2Hkv)Hd, C)
    OTensor Attn_QKV_b;          // (Hq + 2Hkv)Hd
    TTensor Attn_Out_w;      //
    TTensor MLP_Up_w;
    TTensor MLP_Down_w;
};

template<class TTensor>
struct sLLamaNonBlockWeights {
    TTensor Embeddings;      // V, C
    TTensor LMHead;          // V, C
    TTensor LNF_w;           // C
};

template<class TTensor>
struct sLLamaWeightsSet {
    std::vector<sLLamaBlockWeights<TTensor>> Blocks;
    sLLamaNonBlockWeights<TTensor> NonBlocks;
};

struct sLLamaWeights : public ITensorContainer, public sLLamaWeightsSet<TensorShard> {
    virtual ~sLLamaWeights() = default;
    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override;
};

class LLamaWeightsManager : public ITensorContainer {
public:
    virtual ~LLamaWeightsManager();

    static std::unique_ptr<LLamaWeightsManager> create(const LLamaConfig& config, const LLamaOptions& options, int rank, int world, TensorAllocator& alloc);

    void random_init(int seed, const LLamaOptions& options, NCCLCommunicator& comm);
    void import_from_file(const std::string& file_name, bool allow_cast, NCCLCommunicator& comm);
    void export_to_file(const std::string& file_name, NCCLCommunicator& comm) const;

    void synchronize_absmax(NCCLCommunicator& comm);

    void invalidate();
    void reset_scales(cudaStream_t stream);
    std::pair<float*, float*> get_scales_for_block(int layer_idx);

    void begin_optimizer(DeviceMemoryStack& memory, cudaStream_t stream);
    void end_optimizer(DeviceMemoryStack& memory);

    // Weight shards that get updated by the optimizer
    TensorShard& get_master_embeddings();
    TensorShard& get_master_lmhead();
    TensorShard& get_master_lnf_w();

    // In case non-offloaded weights, these functions are trivial no-ops and returns.
    // If master weights are offloaded, then gather initiates the memcpy from host to
    // device buffer (waiting if the buffer is still busy), get returns the buffer,
    // with stream waiting on it being copied. Finally, release copies the buffer back
    // to host using yet another stream (so we get efficient bidirectional communication).

    //! Fetches the master weights from host in case of offloading. Does nothing otherwise.
    //! \param layer_idx The layer for which master weights are requested
    //! \param fetch_stream The stream on which to enqueue the H2D memcpy.
    void fetch_master_block(int layer_idx, cudaStream_t fetch_stream);

    //! Gets the master weights for the given block, making sure stream is blocked in case
    //! they are being fetched from the host.
    //! \param layer_idx The layer for which master weights are requested
    //! \param stream The stream which to block until the preceding gather_master_block has completed.
    sLLamaBlockWeights<TensorShard>& get_master_block(int layer_idx, cudaStream_t stream);

    //! Indicate that the optimizer has finished updating the master weight.
    //! If they are not offloaded, this function does nothing.
    //! Otherwise, if we use quantization and a quant buffer for this layer is available,
    //! quantize the weights while they are still on the device.
    //! Then copy these weights back to host using put_stream, and signal that the buffer can
    //! be reused.
    //! \param layer_idx The layer for which master weights are updated
    //! \param stream The compute stream. Waits on this stream before the master weight is considered updated.
    //! \param put_stream The stream on which to enqueue the H2D memcpy.
    //! \param run_state The run state, needed to run `convert_dtype_for_gather`.
    void release_master_block(int layer_idx, cudaStream_t stream, cudaStream_t put_stream, LLamaRunState& run_state);

    // Weights that will be used during FWD/BWD
    void gather_embeddings(NCCLCommunicator& comm);
    Tensor& get_embeddings(cudaStream_t stream);
    void release_embeddings(cudaStream_t stream);

    void gather_lnf(NCCLCommunicator& comm);
    Tensor& get_lnf(cudaStream_t stream);
    void release_lnf(cudaStream_t stream);

    void gather_head(NCCLCommunicator& comm);
    Tensor& get_head(cudaStream_t stream);
    void release_head(cudaStream_t stream);

    // layers
    void gather_block(int layer_idx, NCCLCommunicator& comm, LLamaRunState& run_state);
    sLLamaBlockWeights<Tensor>& get_block(int layer_idx, cudaStream_t stream);
    void release_block(int layer_idx, cudaStream_t stream);

    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override;
protected:
    LLamaWeightsManager(const LLamaConfig& config, const LLamaOptions& options, int rank, int world);
    void setup_scales(TensorAllocator& alloc);
    void setup_master_buffers(const LLamaConfig& config, TensorAllocator& alloc);

    struct sGatherData {
        int LayerIdx = -1;                  // which layer currently stored in this buffer
        cudaEvent_t DoneEvent = nullptr;    // cuda event to synchronize actions
        bool Fetch = false;                 // indicates whether a gather op has been scheduled
        bool Done = true;                   // indicates whether the param is in use
        int Version = -1;                   // last step at which we gathered this param
    };

    struct sQuantBlock {
        sLLamaBlockWeights<TensorShard> Block;
        int LayerIdx = -1;
        int Version = -1;
    };

    virtual sLLamaBlockWeights<Tensor>& lookup_block_weights(int layer_idx) = 0;
    virtual sQuantBlock& lookup_block_quants(int layer_idx) = 0;
    virtual sGatherData& lookup_block_status(int layer_idx) = 0;

    sLLamaWeights mMaster;
    sLLamaWeightsSet<Tensor> mWork;
    std::vector<sGatherData> mBlockStatus;
    sGatherData mEmbStatus;
    sGatherData mLnfStatus;

    Tensor mAbsMaxes;

    std::array<sLLamaBlockWeights<TensorShard>, 2> mMasterDeviceDoubleBuffer;
    std::array<sGatherData, 2> mMasterDeviceBufferStatus;

    bool is_in_cache(sGatherData& data, int expected) const;
    void update_get_status(sGatherData& data, int expected, cudaStream_t stream) const;
    void release_status(sGatherData& data, int expected, cudaStream_t stream);

    void convert_dtype_for_gather(TensorShard& src, TensorShard& qnt, bool& convert, bool src_is_persistent, LLamaRunState& run_state);

    LLamaConfig mConfig;
    long HQ;    // number of query heads
    long HKV;   // number of key/value heads
    int mShardIdx;
    int mNumShards;

    int mVersion = 0;
    int mHeadID = 0;        // 0 : head == embeddings; 1 : head != embeddings

    ETensorDType mMasterDType;
    ETensorDType mWorkMatDType;

    bool mOffloadMaster;
    bool mUseZeroCopy;
};

sLLamaNonBlockWeights<Tensor> allocate_non_block_full(LLamaConfig config, ETensorDType dtype, EAllocationType kind, TensorAllocator& alloc);
sLLamaNonBlockWeights<TensorShard> allocate_non_block_shard(LLamaConfig config, ETensorDType dtype, EAllocationType kind, int shard_idx, int num_shard, TensorAllocator& alloc);

sLLamaBlockWeights<Tensor> allocate_block_full(const LLamaConfig& config, ETensorDType matrix_dtype, ETensorDType other_dtype, EAllocationType kind, TensorAllocator& alloc);
sLLamaBlockWeights<TensorShard> allocate_block_shard(const LLamaConfig& config, ETensorDType matrix_dtype, ETensorDType other_dtype, EAllocationType kind, int shard_idx, int num_shards, TensorAllocator& alloc);

sLLamaWeightsSet<Tensor> allocate_full_weights(const LLamaConfig& config, EAllocationType kind, TensorAllocator& alloc);
sLLamaWeights allocate_weights(const LLamaConfig& config, EAllocationType kind, int shard_idx, int num_shards, TensorAllocator& alloc);

sLLamaBlockWeights<TensorShard> shard_block(const sLLamaBlockWeights<Tensor>& block, int shard_idx, int num_shards);
sLLamaNonBlockWeights<TensorShard> shard_non_block(const sLLamaNonBlockWeights<Tensor>& block, int shard_idx, int num_shards);

void matrix_params_from_stack(sLLamaBlockWeights<TensorShard>& target, const LLamaConfig& config, ETensorDType dtype, int shard_idx, int num_shards, DeviceMemoryStack& memory);
void non_matrix_params_from_stack(sLLamaBlockWeights<TensorShard>& target, const LLamaConfig& config, ETensorDType dtype, int shard_idx, int num_shards, DeviceMemoryStack& memory);

std::size_t bytes_for_block(const LLamaConfig& config, ETensorDType matrix_dtype, ETensorDType other_dtype, int num_shards);
std::size_t bytes_for_block_matrices(const LLamaConfig& config, ETensorDType dtype, int num_shards);
std::size_t bytes_for_block_non_matrix(const LLamaConfig& config, ETensorDType dtype, int num_shards);

#endif //LLMQ_LLAMA_WEIGHTS_H
