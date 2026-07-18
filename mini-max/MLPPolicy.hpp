#pragma once

#include <array>
#include <string>
#include <vector>

#include "ConnectFourState.hpp"

class MLPPolicy {
public:
    bool load(const std::string& filename, std::string* error_message = nullptr);
    bool isLoaded() const noexcept { return loaded_; }

    std::array<float, 7> predictLogits(const ConnectFourState& state) const;
    int selectAction(const ConnectFourState& state) const;

private:
    bool loaded_ = false;

    // 42 -> 128 -> 64 -> 32 -> 7
    std::vector<float> w1_, b1_;
    std::vector<float> w2_, b2_;
    std::vector<float> w3_, b3_;
    std::vector<float> w4_, b4_;
};
