#pragma once

#include <cstdint>
#include <random>

#include "ConnectFourState.hpp"

class MLPPolicy;
class CNNPolicy;

enum class SearchBudgetMode {
    FixedSimulations,
    FixedTime
};

struct PolicyMCTSConfig {
    SearchBudgetMode budget_mode = SearchBudgetMode::FixedSimulations;
    int simulations = 40;
    int time_limit_ms = 100;
    double c_puct = 1.4;
};

struct PolicyMCTSResult {
    int action = -1;
    int simulations = 0;
    int policy_evaluations = 0;
    std::int64_t policy_inference_ns = 0;
};

PolicyMCTSResult MLPMCTSAction(
    const ConnectFourState& state,
    const MLPPolicy& policy,
    const PolicyMCTSConfig& config,
    std::mt19937& rng
);

PolicyMCTSResult CNNMCTSAction(
    const ConnectFourState& state,
    const CNNPolicy& policy,
    const PolicyMCTSConfig& config,
    std::mt19937& rng
);
