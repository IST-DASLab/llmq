// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace testing_config {

struct TestSizeConfig {
    int B = 2;
    int T = 64;
    int C = 768;
};

inline TestSizeConfig& mutable_cfg() {
    static TestSizeConfig cfg{};
    return cfg;
}

inline void set_test_config(const TestSizeConfig& cfg) {
    mutable_cfg() = cfg;
}

inline const TestSizeConfig& get_test_config() {
    return mutable_cfg();
}

} // namespace testing_config
