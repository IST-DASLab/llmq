// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>

#include "utilities/bounded_cleanup.h"

using namespace std::chrono_literals;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_timed_out_cleanup_retains_exclusive_state_without_blocking_owner() {
    struct Resource {
        std::shared_ptr<std::atomic<int>> destructions;

        explicit Resource(std::shared_ptr<std::atomic<int>> value) : destructions(std::move(value)) {}

        ~Resource() {
            ++*destructions;
        }
    };
    struct State {
        std::unique_ptr<Resource> resource;
        std::shared_ptr<std::atomic<bool>> owner_alive;
        std::shared_ptr<std::atomic<bool>> observed_dead_owner;
        std::shared_future<void> release;
        std::shared_ptr<std::promise<void>> completed;
    };

    auto owner_alive = std::make_shared<std::atomic<bool>>(true);
    auto observed_dead_owner = std::make_shared<std::atomic<bool>>(false);
    auto destructions = std::make_shared<std::atomic<int>>(0);
    auto release = std::make_shared<std::promise<void>>();
    auto completed = std::make_shared<std::promise<void>>();
    auto completion = completed->get_future();
    const auto started = std::chrono::steady_clock::now();

    const auto outcome = run_bounded_cleanup(
        State{std::make_unique<Resource>(destructions), owner_alive,
              observed_dead_owner, release->get_future().share(), completed},
        [](State state) mutable {
            state.release.wait();
            state.observed_dead_owner->store(!state.owner_alive->load());
            state.resource.reset();
            state.completed->set_value();
        },
        20ms);
    owner_alive->store(false);

    require(outcome == BoundedCleanupOutcome::TimedOut, "cleanup did not time out");
    require(std::chrono::steady_clock::now() - started < 500ms, "cleanup blocked its owner");
    release->set_value();
    require(completion.wait_for(500ms) == std::future_status::ready, "detached cleanup did not finish");
    require(observed_dead_owner->load(), "cleanup observed live owner state after timeout");
    require(destructions->load() == 1, "timed-out resource was not destroyed exactly once");
}

void test_completed_cleanup_joins_before_returning() {
    auto cleaned = std::make_shared<std::atomic<int>>(0);
    const auto outcome = run_bounded_cleanup(
        cleaned,
        [](const std::shared_ptr<std::atomic<int>>& value) { ++*value; },
        500ms);

    require(outcome == BoundedCleanupOutcome::Completed, "completed cleanup reported timeout");
    require(cleaned->load() == 1, "cleanup returned before completion");
}

int main() {
    try {
        test_timed_out_cleanup_retains_exclusive_state_without_blocking_owner();
        test_completed_cleanup_joins_before_returning();
        std::cout << "bounded cleanup tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bounded cleanup test failed: " << error.what() << '\n';
        return 1;
    }
}
