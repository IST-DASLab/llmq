// SPDX-License-Identifier: Apache-2.0

#ifndef LLMQ_SRC_UTILITIES_BOUNDED_CLEANUP_H
#define LLMQ_SRC_UTILITIES_BOUNDED_CLEANUP_H

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <utility>

enum class BoundedCleanupOutcome {
    Completed,
    TimedOut,
};

template<class State, class Cleanup>
BoundedCleanupOutcome run_bounded_cleanup(
        State state,
        Cleanup cleanup,
        std::chrono::milliseconds timeout) {
    auto completed = std::make_shared<std::atomic<bool>>(false);
    std::thread worker(
        [state = std::move(state), cleanup = std::move(cleanup), completed]() mutable {
            cleanup(std::move(state));
            completed->store(true, std::memory_order_release);
        });

    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!completed->load(std::memory_order_acquire)
           && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (completed->load(std::memory_order_acquire)) {
        worker.join();
        return BoundedCleanupOutcome::Completed;
    }
    worker.detach();
    return BoundedCleanupOutcome::TimedOut;
}

#endif
