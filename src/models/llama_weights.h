// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_LLAMA_WEIGHTS_H
#define LLMQ_LLAMA_WEIGHTS_H

#include <optional>
#include <vector>

#include "utilities/tensor.h"
#include "utilities/tensor_container.h"

struct LLamaConfig;
struct LLamaOptions;
struct LLamaRunState;
class TensorAllocator;
class NCCLCommunicator;
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

    // Weight shards that get updated by the optimizer
    TensorShard& get_master_embeddings();
    TensorShard& get_master_lmhead();
    TensorShard& get_master_lnf_w();

    void gather_master_block(int layer_idx, cudaStream_t fetch_stream);
    sLLamaBlockWeights<TensorShard>& get_master_block(int layer_idx, cudaStream_t stream);
    void release_master_block(int layer_idx, cudaStream_t stream, cudaStream_t put_stream);

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
        int LayerIdx = -1;
        cudaEvent_t DoneEvent = nullptr;
        bool Fetch = false;
        bool Done = true;
        int Version = -1;
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

    void convert_dtype_for_gather(TensorShard& src, TensorShard& qnt, bool& convert, LLamaRunState& run_state);

    long HQ;    // number of query heads
    long HKV;   // number of key/value heads
    int mShardIdx;
    int mNumShards;

    int mVersion = 0;
    int mHeadID = 0;        // 0 : head == embeddings; 1 : head != embeddings

    ETensorDType mMasterDType;
    ETensorDType mWorkMatDType;

    bool mOffloadMaster;
};

sLLamaNonBlockWeights<Tensor> allocate_non_block_full(LLamaConfig config, ETensorDType dtype, EAllocationType kind, TensorAllocator& alloc);
sLLamaNonBlockWeights<TensorShard> allocate_non_block_shard(LLamaConfig config, ETensorDType dtype, EAllocationType kind, int shard_idx, int num_shard, TensorAllocator& alloc);

sLLamaBlockWeights<Tensor> allocate_block_full(const LLamaConfig& config, ETensorDType matrix_dtype, ETensorDType other_dtype, EAllocationType kind, TensorAllocator& alloc);
sLLamaBlockWeights<TensorShard> allocate_block_shard(const LLamaConfig& config, ETensorDType matrix_dtype, ETensorDType other_dtype, EAllocationType kind, int shard_idx, int num_shards, TensorAllocator& alloc);

sLLamaWeightsSet<Tensor> allocate_full_weights(const LLamaConfig& config, EAllocationType kind, TensorAllocator& alloc);
sLLamaWeights allocate_weights(const LLamaConfig& config, EAllocationType kind, int shard_idx, int num_shards, TensorAllocator& alloc);

sLLamaBlockWeights<TensorShard> shard_block(const sLLamaBlockWeights<Tensor>& block, int shard_idx, int num_shards);
sLLamaNonBlockWeights<TensorShard> shard_non_block(const sLLamaNonBlockWeights<Tensor>& block, int shard_idx, int num_shards);

std::size_t bytes_for_block(const LLamaConfig& config, ETensorDType matrix_dtype, ETensorDType other_dtype, int num_shards);
std::size_t bytes_for_block_matrices(const LLamaConfig& config, ETensorDType dtype, int num_shards);
std::size_t bytes_for_block_non_matrix(const LLamaConfig& config, ETensorDType dtype, int num_shards);

#endif //LLMQ_LLAMA_WEIGHTS_H
