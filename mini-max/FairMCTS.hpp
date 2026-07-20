#pragma once

#include <cstdint>
#include <random>

#include "ConnectFourState.hpp"

class MLPPolicy;
class CNNPolicy;

enum class FairSearchBudgetMode {
    FixedSimulations,
    FixedTime
};

struct FairMCTSConfig {
    FairSearchBudgetMode budget_mode =
        FairSearchBudgetMode::FixedSimulations;
    int simulations = 10000;
    int time_limit_ms = 100;

    double uct_c = 1.41421356237;
    double root_policy_weight = 0.80;
    double uniform_mix = 0.40;
};

struct FairMCTSResult {
    int action = -1;
    int simulations = 0;
    int policy_evaluations = 0;
    std::int64_t policy_inference_ns = 0;
};

FairMCTSResult PlainMCTSAction(
    const ConnectFourState& state,
    const FairMCTSConfig& config,
    std::mt19937& rng
);

FairMCTSResult MLPRootMCTSAction(
    const ConnectFourState& state,
    const MLPPolicy& policy,
    const FairMCTSConfig& config,
    std::mt19937& rng
);

FairMCTSResult CNNRootMCTSAction(
    const ConnectFourState& state,
    const CNNPolicy& policy,
    const FairMCTSConfig& config,
    std::mt19937& rng
);
