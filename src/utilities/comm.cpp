// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "comm.h"

#include <stdexcept>
#include <utility>
#include <variant>
#include <future>

#include <nccl.h>
#include <fmt/core.h>

#include "gpu_info.h"
#include "kernels/kernels.h"
#include "tensor.h"
#include "utils.h"

void nccl_check(ncclResult_t status, const char* file, int line) {
    if (status != ncclSuccess) {
        throw std::runtime_error(fmt::format("NCCL error at {}:{}: {}", file, line, ncclGetErrorString(status)));
    }
}
#define ncclCheck(err) (nccl_check(err, __FILE__, __LINE__))

struct NCCLCommunicator::CommandBuffer
{
    struct Gather {
        std::byte* Src;
        std::byte* Dst;
        std::size_t Bytes;
    };

    struct ScatterReduce {
        ETensorDType DType;
        std::byte* Tensor;
        std::size_t Elements;
    };


    struct Send {
        const std::byte* Tensor;
        std::size_t Bytes;
        int Target;
    };

    struct Recv {
        std::byte* Tensor;
        std::size_t Bytes;
        int Source;
    };

    std::vector<std::variant<Gather, ScatterReduce, Send, Recv>> Commands;
    cudaEvent_t Ready = nullptr;
};

NCCLCommunicator::NCCLCommunicator(int rank, int world, const void* nccl_id) :
    mRank(rank), mWorld(world), mNcclComm(nullptr), mCmdBuf(std::make_unique<CommandBuffer>())
{
    CUDA_CHECK(cudaSetDevice(mRank));
    ncclCheck(ncclCommInitRank(&mNcclComm, mWorld, *reinterpret_cast<const ncclUniqueId*>(nccl_id), mRank));

    // must be created _after_ we set the device
    mCommsStream = create_named_stream("nccl_stream");
    mCommsSync = create_named_event("nccl_sync");  // todo disable timing for max perf
}

#include <pthread.h>

NCCLCommunicator::~NCCLCommunicator() {
    // When used with the python bindings, ncclCommFinalize() can hang forever;
    // I haven't found a fix, so here we just make sure that the hang gets localized
    // to a helper thread (which we leak, but generally ~NCCLCommunicator is expected
    // to run at program shutdown anyway)
    auto terminate_future = std::async(std::launch::async, [this]() {
        this->terminate_nccl();
    });

    if (terminate_future.wait_for(std::chrono::seconds(2)) == std::future_status::timeout) {
        fprintf(stderr, "NCCL termination timed out, detaching\n");
        // this *will* leak resources, but at least we're not hanging forever
        new auto(std::move(terminate_future));
    }
    CUDA_CHECK(cudaEventDestroy(mCommsSync));
    CUDA_CHECK(cudaStreamDestroy(mCommsStream));
}

void NCCLCommunicator::terminate_nccl() {
    ncclResult_t result;
    ncclCheck(ncclCommGetAsyncError(mNcclComm, &result));
    // do "nice" shutdown if we're in a good state,
    // just abort if there is something weird going on.
    if (std::uncaught_exceptions() == 0 && result == ncclSuccess) {
        CUDA_CHECK(cudaStreamSynchronize(mCommsStream));
        CUDA_CHECK(cudaDeviceSynchronize());
        ncclCheck(ncclCommFinalize(mNcclComm));
        ncclCheck(ncclCommDestroy(mNcclComm));
    } else {
        ncclCheck(ncclCommAbort(mNcclComm));
    }
}

void NCCLCommunicator::begin_transaction(cudaEvent_t ready) {
    if (!mCmdBuf->Commands.empty()) {
        throw std::runtime_error("start_comms: Buffer not empty");
    }
    mCmdBuf->Ready = ready;
}

void NCCLCommunicator::begin_transaction(cudaStream_t wait_for_stream) {
    CUDA_CHECK(cudaEventRecord(mCommsSync, wait_for_stream));
    begin_transaction(mCommsSync);
}

struct NCCLCommunicator::CommandVisitor {
    NCCLCommunicator* Comm;

    void operator()(CommandBuffer::Gather& cmd) const {
        Comm->gather_weight(cmd.Src, cmd.Dst, cmd.Bytes);
    }

    void operator()(CommandBuffer::ScatterReduce& cmd) const {
        switch (cmd.DType) {
        case ETensorDType::FP32:
            Comm->scatter_grad(reinterpret_cast<float*>(cmd.Tensor), cmd.Elements);
            break;
        case ETensorDType::BF16:
            Comm->scatter_grad(reinterpret_cast<nv_bfloat16*>(cmd.Tensor), cmd.Elements);
            break;
        default:
            throw std::runtime_error("scatter: Unsupported dtype");
        }
    }

    void operator()(CommandBuffer::Send& cmd) const {
        Comm->send(cmd.Tensor, cmd.Target, cmd.Bytes);
    }

    void operator()(CommandBuffer::Recv& cmd) const {
        Comm->recv(cmd.Tensor, cmd.Source, cmd.Bytes);
    }

};

void NCCLCommunicator::execute_transaction(cudaEvent_t signal) {
    _launch_queue_throttle_sync();

    on_execute_transaction(*mCmdBuf);

    CommandVisitor visitor{this};
    for (auto& cmd: mCmdBuf->Commands) {
        std::visit(visitor, cmd);
    }

    on_finish_transaction(signal);

    // make sure no GPU can enqueue new work until *all* GPUs have enqueued this transaction
    // this prevents a faster process from filling up the launch queue, which could block
    // a slower process when it tried to enqueue work that this transaction depends on, causing a
    // deadlock
    _launch_queue_throttle_sync();

    mCmdBuf->Commands.clear();
}

void NCCLCommunicator::schedule_reduce_scatter(Tensor& tensor) {
    if (tensor.Data == nullptr) {
        throw std::runtime_error("scatter: Source tensor is null");
    }

    mCmdBuf->Commands.emplace_back(CommandBuffer::ScatterReduce{.DType = tensor.DType, .Tensor = tensor.Data, .Elements = tensor.nelem()});
}

void NCCLCommunicator::schedule_all_gather(const TensorShard& src, Tensor& tgt) {
    if (src.Data == nullptr) {
        throw std::runtime_error("gather: Source tensor is null");
    }

    if (tgt.Data == nullptr) {
        throw std::runtime_error("gather: Target tensor is null");
    }

    if (src.DType != tgt.DType) {
        throw std::runtime_error("gather: Mismatched dtypes");
    }

    mCmdBuf->Commands.emplace_back(CommandBuffer::Gather{.Src = src.Data, .Dst = tgt.Data, .Bytes = tgt.bytes()});
}

void NCCLCommunicator::reduce_loss(float* loss, cudaStream_t stream) {
    ncclCheck(ncclAllReduce(loss, loss, 1, ncclFloat, ncclAvg, mNcclComm, stream));
}

void NCCLCommunicator::reduce_max(float* values, int n, cudaStream_t stream) {
    ncclCheck(ncclAllReduce(values, values, n, ncclFloat, ncclMax, mNcclComm, stream ? stream : mCommsStream));
}

void NCCLCommunicator::reduce_norm(float* norm_squared, cudaStream_t stream) {
    ncclCheck(ncclAllReduce(norm_squared, norm_squared, 1, ncclFloat, ncclSum, mNcclComm, stream));
}

void NCCLCommunicator::scatter_grad(float* value, std::size_t size) {
    assert(size % mWorld == 0);
    size_t shard_size = size / mWorld;
    ptrdiff_t shard_offset = (ptrdiff_t)shard_size * mRank;
    ncclCheck(ncclReduceScatter(
        value, value + shard_offset,
        shard_size,
        ncclFloat, ncclAvg,
        mNcclComm, mCommsStream
    ));
}

void NCCLCommunicator::scatter_grad(nv_bfloat16* value, std::size_t size) {
    assert(size % mWorld == 0);
    size_t shard_size = size / mWorld;
    ptrdiff_t shard_offset = (ptrdiff_t)shard_size * mRank;
    ncclCheck(ncclReduceScatter(
        value, value + shard_offset,
        shard_size,
        ncclBfloat16, ncclAvg,
        mNcclComm, mCommsStream
    ));
}

void NCCLCommunicator::gather_weight(const std::byte* src, std::byte* dst,  std::size_t size) {
    assert(size % mWorld == 0);
    size_t shard_size = size / mWorld;
    if(src == dst) {
        src += shard_size * mRank; // in-place
    }
    ncclCheck(ncclAllGather(src,
                            dst,
                            shard_size, ncclInt8,
                            mNcclComm, mCommsStream));
}

void NCCLCommunicator::send(const std::byte* src, int peer, std::size_t size) {
    ncclCheck(ncclSend(src, size, ncclInt8, peer, mNcclComm, mCommsStream));
}

void NCCLCommunicator::recv(std::byte* dst, int peer, std::size_t size) {
    ncclCheck(ncclRecv(dst, size, ncclInt8, peer, mNcclComm, mCommsStream));
}

void NCCLCommunicator::wait_on_comms(cudaStream_t compute_stream) {
    CUDA_CHECK(cudaStreamWaitEvent(compute_stream, mCommsSync, 0));
}

#if USE_MPI

// macro conflict :(
#undef HOST
#include <mpi.h>

void mpi_check(int status, const char *file, int line) {
    if (status != MPI_SUCCESS) {
        char mpi_error[4096];
        int mpi_error_len = 0;
        if(MPI_Error_string(status, &mpi_error[0], &mpi_error_len) == MPI_SUCCESS) {
            throw std::runtime_error(fmt::format("Failed to create MPI error string for error at {}:{} ({})", file, line, status));
        }
        throw std::runtime_error(fmt::format("MPI error at {}:{}: {}", file, line, mpi_error));
    }
}
#define mpiCheck(err) (mpi_check(err, __FILE__, __LINE__))

class NCCLCommunicatorMPI : public NCCLCommunicator {
public:
    using NCCLCommunicator::NCCLCommunicator;
    ~NCCLCommunicatorMPI() override;
    void barrier() override;
    void _launch_queue_throttle_sync() override {};  // not needed with separate processes
    void gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) override;
    void all_gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) override;

    void on_execute_transaction(const NCCLCommunicator::CommandBuffer& cmd) override;
    void on_finish_transaction(cudaEvent_t signal) override;
};


NCCLCommunicatorMPI::~NCCLCommunicatorMPI() {
    int is_init = 0;
    mpiCheck(MPI_Initialized(&is_init));
    // I've observed that (at least in some circumstances), when
    // an exception is active, MPI_Finalize just blocked forever...
    if(is_init && std::uncaught_exceptions() == 0) {
        mpiCheck(MPI_Finalize());
    }
}

void NCCLCommunicatorMPI::barrier() {
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
}

void NCCLCommunicatorMPI::gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) {
    mpiCheck(MPI_Gather(object, size, MPI_BYTE, recv, size, MPI_BYTE, 0, MPI_COMM_WORLD));
}

void NCCLCommunicatorMPI::all_gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) {
    mpiCheck(MPI_Allgather(object, size, MPI_BYTE, recv, size, MPI_BYTE, MPI_COMM_WORLD));
}

void NCCLCommunicatorMPI::on_execute_transaction(const NCCLCommunicator::CommandBuffer& cmd) {
    CUDA_CHECK(cudaStreamWaitEvent(stream(), cmd.Ready));
    ncclCheck(ncclGroupStart());
}

void NCCLCommunicatorMPI::on_finish_transaction(cudaEvent_t signal) {
    ncclCheck(ncclGroupEnd());
    CUDA_CHECK(cudaEventRecord(signal, stream()));
}

std::unique_ptr<NCCLCommunicator> NCCLCommunicator::make_mpi_communicator() {
    mpiCheck(MPI_Init(nullptr, nullptr));
    int rank, world;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &world));

    ncclUniqueId nccl_id;
    if (rank == 0) {
        ncclCheck(ncclGetUniqueId(&nccl_id));
    }
    mpiCheck(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));

    return std::make_unique<NCCLCommunicatorMPI>(rank, world, &nccl_id);
}

#else
std::unique_ptr<NCCLCommunicator> NCCLCommunicator::make_mpi_communicator() {
    throw std::runtime_error("MPI communicator not available.");
}


#endif

#if USE_THREADS

#include <thread>
#include <barrier>

class NCCLCommunicatorThreads : public NCCLCommunicator {
public:
    struct SharedState {
        std::unique_ptr<std::barrier<>> Barrier;
        std::vector<std::byte*> Buffer;     // one pointer per thread
        std::vector<std::exception_ptr> Exceptions;
        std::mutex Mutex;
    };

    NCCLCommunicatorThreads(int rank, int world, bool memcpy_allgather, bool memcpy_send_recv, const void* nccl_id, std::shared_ptr<SharedState> state);
    ~NCCLCommunicatorThreads() override;
    void barrier() override;
    void _launch_queue_throttle_sync() override;
    void gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) override;
    void all_gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) override;

    void gather_weight(const std::byte* src, std::byte* tgt, std::size_t size) override;
    void send(const std::byte* src, int peer, std::size_t size) override;
    void recv(std::byte* tgt, int peer, std::size_t size) override;

    void on_execute_transaction(const NCCLCommunicator::CommandBuffer& cmd) override;
    void on_finish_transaction(cudaEvent_t signal) override;

private:
    std::shared_ptr<SharedState> mShare;
    bool mAllGatherUseMemcpy = false;
    bool mSendRecvUseMemcpy = true;

    // transaction status
    bool mUseMemcpy;
    bool mUseNCCL;

    struct sSendParams {
        const std::byte* Data;
        std::size_t Size;
        int Peer;
        bool Matched = false;
    };
    std::vector<sSendParams> mSendParams;

    struct sRecvParams {
        std::byte* Data;
        std::size_t Size;
        int Peer;
    };
    std::vector<sRecvParams> mRecvParams;
};

NCCLCommunicatorThreads::NCCLCommunicatorThreads(int rank, int world, bool memcpy_allgather, bool memcpy_send_recv, const void* nccl_id, std::shared_ptr<SharedState> state):
    NCCLCommunicator(rank, world, nccl_id), mShare(std::move(state)), mAllGatherUseMemcpy(memcpy_allgather), mSendRecvUseMemcpy(memcpy_send_recv) {
}

NCCLCommunicatorThreads::~NCCLCommunicatorThreads() {
    if(mShare && mShare->Barrier) {
        mShare->Barrier->arrive_and_drop();
    }
}

class ThreadsPackImp : public CommunicatorThreadsPack {
public:
    ThreadsPackImp(std::vector<std::jthread> threads, std::shared_ptr<NCCLCommunicatorThreads::SharedState> state) :
        mThreads(std::move(threads)), mState(std::move(state)){

    }

    ~ThreadsPackImp() override {
        join_impl();
    }

    void join() override {
        join_impl();
    }

    bool has_exception() const override {
        std::lock_guard<std::mutex> lock(mState->Mutex);
        for(int t = 0; t < mThreads.size(); ++t) {
            if (auto error = mState->Exceptions[t]; error) {
                return true;
            }
        }
        return false;
    }
private:
    void join_impl() {
        // if any worker thread has already crashed, raise that exception in the main thread
        check_exceptions();

        for(auto& t: mThreads) {
            if(t.joinable()) {
                t.join();
            }
        }

        // ok, now that everyone has terminated, check again for proper exit
        check_exceptions();
    }

    void check_exceptions() {
        std::lock_guard<std::mutex> lock(mState->Mutex);
        for(int t = 0; t < mThreads.size(); ++t) {
            if(auto error = mState->Exceptions[t]; error) {
                fprintf(stderr, "Thread %d exited with uncaught exception\n", t);
                fflush(stderr);
                // reset the exception and rethrow it
                mState->Exceptions[t] = nullptr;
                std::rethrow_exception(error);
            }
        }
    }

    std::vector<std::jthread> mThreads;
    std::shared_ptr<NCCLCommunicatorThreads::SharedState> mState;
};

void NCCLCommunicator::run_threads_communicators(int ngpus, bool memcpy_allgather, bool memcpy_send_recv, std::function<void(NCCLCommunicator& comm)> work) {
    auto threads = launch_threads_communicators(ngpus, memcpy_allgather, memcpy_send_recv, std::move(work));
    threads->join();
}

std::unique_ptr<CommunicatorThreadsPack> NCCLCommunicator::launch_threads_communicators(
            int ngpus, bool memcpy_allgather, bool memcpy_send_recv, std::function<void(NCCLCommunicator& comm)> work)
{
    std::vector<std::jthread> threads;
    ncclUniqueId nccl_id;
    ncclCheck(ncclGetUniqueId(&nccl_id));
    threads.reserve(ngpus);
    auto bar = std::make_shared<NCCLCommunicatorThreads::SharedState>(std::make_unique<std::barrier<>>(ngpus), std::vector<std::byte*>(ngpus));
    bar->Exceptions.resize(ngpus);
    for(int i = 0; i < ngpus; ++i) {
        threads.emplace_back([i, ngpus, nccl_id, memcpy_allgather, memcpy_send_recv, work, bar]() {
            try {
                if (!set_cpu_affinity()) {
                    fprintf(stderr, "WARNING: Failed to set CPU affinity for rank %d\n", i);
                }
                NCCLCommunicatorThreads comm(i, ngpus, memcpy_allgather, memcpy_send_recv, &nccl_id, bar);
                work(comm);
                bar->Barrier->arrive_and_wait();
            } catch(...) {
                std::lock_guard<std::mutex> lock(bar->Mutex);
                bar->Exceptions[i] = std::current_exception();
            }
        }
        );
    }
    return std::make_unique<ThreadsPackImp>(std::move(threads), std::move(bar));
}

void NCCLCommunicator::schedule_destructive_all_to_all(const Tensor& tensor) {
    std::size_t shard_size = (ptrdiff_t)tensor.bytes() / world_size();
    for(int n = 1; n < world_size(); ++n) {
        int dst = (n + rank()) % world_size();
        int src = (rank() - n + world_size()) % world_size();
        int store = (rank() + n - 1 + world_size()) % world_size();
        mCmdBuf->Commands.emplace_back(CommandBuffer::Send{
            .Tensor = tensor.Data + dst * shard_size,
            .Bytes = shard_size,
            .Target = dst
            }
            );
        mCmdBuf->Commands.emplace_back(CommandBuffer::Recv{
            .Tensor = tensor.Data + store * shard_size,
            .Bytes = shard_size,
            .Source = src
        });
    }
}

void NCCLCommunicatorThreads::send(const std::byte* src, int peer, std::size_t size) {
    if (!mSendRecvUseMemcpy) {
        NCCLCommunicator::send(src, peer, size);
    } else {
        mSendParams.emplace_back(sSendParams{src, size, peer});
    }
}

void NCCLCommunicatorThreads::recv(std::byte* tgt, int peer, std::size_t size) {
    if (!mSendRecvUseMemcpy) {
        NCCLCommunicator::recv(tgt, peer, size);
    } else {
        mRecvParams.emplace_back(sRecvParams{tgt, size, peer});
    }
}

void NCCLCommunicatorThreads::gather_weight(const std::byte* src, std::byte* tgt, std::size_t size) {
    if(mAllGatherUseMemcpy) {
        auto wgt_list = host_all_gather(src);
        std::size_t shard_size = size / world_size();
        for (int i = 0; i < world_size(); ++i) {
            if (tgt + shard_size * i != wgt_list[i]) {
                CUDA_CHECK(cudaMemcpyAsync(tgt + shard_size * i, wgt_list[i], shard_size, cudaMemcpyDeviceToDevice, stream()));
            }
        }
    } else {
        NCCLCommunicator::gather_weight(src, tgt, size);
    }
}

void NCCLCommunicatorThreads::on_execute_transaction(const NCCLCommunicator::CommandBuffer& commands) {
    mUseMemcpy = false;
    mUseNCCL = false;
    for (auto& cmd: commands.Commands) {
        if (std::holds_alternative<CommandBuffer::ScatterReduce>(cmd)) {
            mUseNCCL = true;
        }
        if (std::holds_alternative<CommandBuffer::Gather>(cmd)) {
            if (!mAllGatherUseMemcpy) mUseNCCL = true;
            if (mAllGatherUseMemcpy) mUseMemcpy = true;
        }
        if (std::holds_alternative<CommandBuffer::Send>(cmd)) {
            if (!mSendRecvUseMemcpy) mUseNCCL = true;
            if (mSendRecvUseMemcpy) mUseMemcpy = true;
        }
    }

    assert(mUseNCCL || mUseMemcpy);

    if(mUseMemcpy) {
        // ensure every worker has set-up commands.Ready to the most recent version
        barrier();
        // get the ready event from all workers
        auto event_list = host_all_gather(commands.Ready);
        // make sure to block the comms thread until the data is ready on every worker
        for (auto event: event_list) {
            CUDA_CHECK(cudaStreamWaitEvent(stream(), event, 0));
        }
    }

    if(mUseNCCL){
        CUDA_CHECK(cudaStreamWaitEvent(stream(), commands.Ready, 0));
        ncclCheck(ncclGroupStart());
    }
}

void NCCLCommunicatorThreads::on_finish_transaction(cudaEvent_t signal) {
    if (!mRecvParams.empty()) {
        // get send-queues from peers
        std::vector<std::vector<sSendParams>*> send_params = host_all_gather(&mSendParams);
        std::vector<cudaEvent_t> sync_events = host_all_gather(signal);
        // ok, now iterate all recv's
        for (auto& recv: mRecvParams) {
            // find matching send
            for (auto& send : *send_params.at(recv.Peer)) {
                if (send.Peer != rank() || send.Matched) continue;
                // copy data
                if (recv.Size != send.Size) {
                    throw std::runtime_error("Size mismatch in recv/send");
                }
                CUDA_CHECK(cudaMemcpyAsync(recv.Data, send.Data, recv.Size, cudaMemcpyDeviceToDevice, stream()));
                send.Matched = true;
                break;
            }

            CUDA_CHECK(cudaEventRecord(signal, stream()));
            barrier();      // assumes _all_ workers have the same number of receives!
            for (int j = 0; j < world_size(); ++j) {
                if (j != rank()) {
                    CUDA_CHECK(cudaStreamWaitEvent(stream(), sync_events[j], 0));
                }
            }
        }

        barrier();
        mRecvParams.clear();
        mSendParams.clear();
    }
    if(mUseNCCL) {
        ncclCheck(ncclGroupEnd());
    }

    CUDA_CHECK(cudaEventRecord(signal, stream()));
}

void NCCLCommunicatorThreads::barrier() {
    mShare->Barrier->arrive_and_wait();
}

void NCCLCommunicatorThreads::_launch_queue_throttle_sync() {
    // This is *speculation* on my part, based on the observation that I was getting deadlocks
    // in multi-threaded, but not in multi-process configuration, and the deadlock would go away
    // using either cudaDeviceSynchronize (so no sync between threads, only GPU to CPU)
    // or barrier (so only syncing CPU threads, no GPU sync). My suspicion is that there is some
    // form of a *per-process* launch queue, and if one thread runs ahead and fills it up too much,
    // the other thread cannot schedule the kernels that the first thread is waiting for.
    // So, for multithreaded mode, let's just ensure that CPU threads do synchronize regularly
    this->barrier();
}

void NCCLCommunicatorThreads::gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) {
    if(rank() == 0) {
        mShare->Buffer[0] = recv;
    }
    barrier();
    std::memcpy(mShare->Buffer[0] + rank() * size, object, size);
    barrier();
    if(rank() == 0) {
        mShare->Buffer[0] = nullptr;
    }
}

void NCCLCommunicatorThreads::all_gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) {
    barrier();
    mShare->Buffer[rank()] = const_cast<std::byte*>(object);
    barrier();
    for(int i = 0; i < world_size(); ++i) {
        std::memcpy(recv + i * size,  mShare->Buffer[i], size);
    }
    barrier();
    mShare->Buffer[rank()] = nullptr;
}
#else
void NCCLCommunicator::launch_threads_communicators(int zero_level) {
    throw std::runtime_error("threads communicator not available.");
}

#endif
