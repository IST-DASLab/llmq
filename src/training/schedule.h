// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
//
// SPDX-License-Identifier: MIT

#ifndef LLMQ_SRC_UTILITIES_SCHEDULE_H
#define LLMQ_SRC_UTILITIES_SCHEDULE_H

#include <cmath>
#include <numbers>

class LRSchedule {
public:
    explicit LRSchedule(float peak_rate) : mPeakRate(peak_rate), mBaseRate(peak_rate) {}
    LRSchedule(float peak_rate, int steps, int warmup, float base_rate) : mPeakRate(peak_rate), mDecaySteps(steps), mWarmupSteps(warmup), mBaseRate(base_rate) {}
    float get_lr(int step) {
        if(step < mWarmupSteps) {
            return mPeakRate * step / mWarmupSteps;
        }
        double pos = (double)(step - mWarmupSteps) / mDecaySteps;
        double frac = 0.5 * std::cos(pos * std::numbers::pi) + 0.5;
        return static_cast<float>(frac * (mPeakRate - mBaseRate) + mBaseRate);
    }
private:
    int mWarmupSteps = 0;
    int mDecaySteps = 1;
    float mPeakRate;
    float mBaseRate;
};

#endif //LLMQ_SRC_UTILITIES_SCHEDULE_H
