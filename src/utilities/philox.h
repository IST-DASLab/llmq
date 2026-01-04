// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_UTILS_PHILOX_H
#define LLMQ_SRC_UTILS_PHILOX_H

#include <array>
#include <span>
#include <cstdint>
#include <utility>

class Philox4x32 {
private:
    static constexpr std::uint32_t MC[4] = {0xD2511F53, 0x9E3779B9, 0xCD9E8D57, 0xBB67AE85};
    static constexpr int ROUNDS = 10;

    uint32_t key[2];

    static std::pair<std::uint32_t, std::uint32_t> mul_hilo(std::uint32_t a, std::uint32_t b) {
        std::uint64_t ab = static_cast<std::int64_t>(a) * static_cast<std::int64_t>(b);
        return {static_cast<std::uint32_t>(ab >> 32ull), static_cast<std::uint32_t>(ab & 0x00000000ffffffffull)};
    }

public:
    explicit Philox4x32(uint64_t seed = 1) {
        key[0] = static_cast<uint32_t>(seed);
        key[1] = static_cast<uint32_t>(seed >> 32);
    }

    Philox4x32(uint32_t seed_a, std::uint32_t seed_b) : key{seed_a, seed_b} {
    }

    void generate_4(std::span<std::uint32_t, 4> dst, std::uint32_t x, std::uint32_t y) {
        // Extract counter parts
        uint32_t R0 = x;
        uint32_t L0 = y;

        uint32_t R1 = 0;  // High part of counter for 4x32
        uint32_t L1 = 0;

        // Initialize keys
        uint32_t K0 = key[0];
        uint32_t K1 = key[1];

        // Perform rounds
        for (int i = 0; i < ROUNDS; ++i) {
            auto [hi0, lo0] = mul_hilo(R0, MC[0]);
            auto [hi1, lo1] = mul_hilo(R1, MC[2]);

            R0 = hi1 ^ L0 ^ K0;
            L0 = lo1;
            R1 = hi0 ^ L1 ^ K1;
            L1 = lo0;

            K0 = (K0 + MC[1]) & 0xFFFFFFFF;
            K1 = (K1 + MC[3]) & 0xFFFFFFFF;
        }

        dst[0] = R0;
        dst[1] = L0;
        dst[2] = R1;
        dst[3] = L1;
    }

    template<std::size_t S>
    void generate(std::span<std::uint32_t, S> dst, std::uint32_t x, std::uint32_t y) {
        static_assert(S % 4 == 0, "philox generates numbers in groups of 4");
        for(std::size_t i = 0; i < S; i += 4) {
            generate_4(std::span<std::uint32_t, 4>(dst.data() + i, 4), x + static_cast<std::uint32_t>(i / 4), y);
        }
    }

    // Generate 4 random 32-bit values
    std::array<uint32_t, 4> generate(std::uint32_t x, std::uint32_t y) {
        std::array<uint32_t, 4> res;
        generate_4(res, x, y);
        return res;
    }
};

#endif //LLMQ_SRC_UTILS_PHILOX_H
