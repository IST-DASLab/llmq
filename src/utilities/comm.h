// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_UTILITIES_COMM_H
#define LLMQ_SRC_UTILITIES_COMM_H

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include <cuda_bf16.h>

namespace std
{
    class jthread;
}

struct Tensor;
struct TensorShard;

typedef struct ncclComm* ncclComm_t;
typedef struct CUevent_st* cudaEvent_t;
typedef struct CUstream_st* cudaStream_t;
class NCCLCommunicator;

class CommunicatorThreadsPack {
public:
    virtual ~CommunicatorThreadsPack() = default;
    virtual void join() = 0;
    virtual bool has_exception() const = 0;
};


class NCCLCommunicator {
public:
    NCCLCommunicator(int rank, int world, const void* nccl_id);
    virtual ~NCCLCommunicator();

    // Cpu-side barrier
    virtual void barrier() = 0;

    void begin_transaction(cudaEvent_t ready);
    void begin_transaction(cudaStream_t wait_for_stream);
    void schedule_reduce_scatter(Tensor& tensor);
    void schedule_all_gather(const TensorShard& src, Tensor& tgt);
    // like all-to-all, except the local shard will *not* be preserved, and results will be shifted cyclically
    void schedule_destructive_all_to_all(const Tensor& tensor);
    void execute_transaction(cudaEvent_t signal);

    void reduce_loss(float* loss, cudaStream_t stream);
    void reduce_norm(float* norm_squared, cudaStream_t stream);

    void reduce_max(float* values, int n = 1, cudaStream_t stream=nullptr);

    void wait_on_comms(cudaStream_t compute_stream);

    [[nodiscard]] int rank() const { return mRank; }
    [[nodiscard]] int world_size() const { return mWorld; }

    [[nodiscard]] cudaStream_t stream() const { return mCommsStream; }

    //! On the root rank, returns a vector of (memcpyable) T objects that
    //! have been gathered from all ranks.
    template<typename T>
    std::vector<T> host_gather(const T& object) {
        static_assert(std::is_trivially_copyable_v<T>, "Cannot communicate type with non-trivial copy operator");
        std::vector<T> result;
        if(rank() == 0) {
            result.resize(world_size());
        }

        gather_bytes_host(reinterpret_cast<std::byte*>(result.data()), reinterpret_cast<const std::byte*>(&object), sizeof(T));
        return result;
    }

    template<typename T>
    std::vector<T> host_all_gather(const T& object) {
        static_assert(std::is_trivially_copyable_v<T>, "Cannot communicate type with non-trivial copy operator");
        std::vector<T> result(world_size());
        all_gather_bytes_host(reinterpret_cast<std::byte*>(result.data()), reinterpret_cast<const std::byte*>(&object), sizeof(T));
        return result;
    }

    static std::unique_ptr<NCCLCommunicator> make_mpi_communicator();
    static void run_threads_communicators(int ngpus, bool memcpy_allgather, bool memcpy_send_recv, std::function<void(NCCLCommunicator& comm)> work);
    static std::unique_ptr<CommunicatorThreadsPack> launch_threads_communicators(int ngpus, bool memcpy_allgather, bool memcpy_send_recv, std::function<void(NCCLCommunicator& comm)> work);
protected:
    void terminate_nccl();

    void scatter_grad(float* value, std::size_t size);
    void scatter_grad(nv_bfloat16* value, std::size_t size);

    virtual void gather_weight(const std::byte* src, std::byte* tgt, std::size_t size);
    virtual void send(const std::byte* src, int peer, std::size_t size);
    virtual void recv(std::byte* tgt, int peer, std::size_t size);


    virtual void gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) = 0;
    virtual void all_gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) = 0;

    struct CommandBuffer;
    virtual void on_execute_transaction(const CommandBuffer&) = 0;
    virtual void on_finish_transaction(cudaEvent_t signal) = 0;
    virtual void _launch_queue_throttle_sync() = 0;
private:
    ncclComm_t mNcclComm;
    int mRank;
    int mWorld;

    cudaEvent_t mCommsSync;
    cudaStream_t mCommsStream;

    struct CommandVisitor;
    std::unique_ptr<CommandBuffer> mCmdBuf;

    friend struct CommandVisitor;
};

#endif //LLMQ_SRC_UTILITIES_COMM_H
