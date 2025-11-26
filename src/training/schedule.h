// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_TRAINING_SCHEDULE_H
#define LLMQ_SRC_TRAINING_SCHEDULE_H

#include <cmath>
#include <numbers>

class ISchedule {
public:
    virtual ~ISchedule() = default;
    virtual float eval(int step) const = 0;
};

class CosineSchedule : public ISchedule {
public:
    explicit CosineSchedule(float peak_rate) : mPeakRate(peak_rate), mBaseRate(peak_rate) {}
    CosineSchedule(float peak_rate, int steps, int warmup, float base_rate) : mPeakRate(peak_rate), mDecaySteps(steps), mWarmupSteps(warmup), mBaseRate(base_rate) {}
    float eval(int step) const override {
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

class LinearSchedule : public ISchedule {
public:
    explicit LinearSchedule(float start, float end, int steps, int warmup) : mStart(start), mEnd(end), mDecaySteps(steps), mWarmupSteps(warmup) {  }

    float eval(int step) const override {
        if(step < mWarmupSteps) {
            return mStart * step / mWarmupSteps;
        }
        double pos = (double)(step - mWarmupSteps) / mDecaySteps;
        pos = std::clamp(pos, 0.0, 1.0);
        return mStart + (mEnd-mStart) * pos;
    }
private:
    int mWarmupSteps = 0;
    float mStart;
    float mEnd;
    int mDecaySteps;
};

#endif //LLMQ_SRC_TRAINING_SCHEDULE_H
