#pragma once

#include <array>
#include <string>
#include <vector>

#include "ConnectFourState.hpp"

class CNNPolicy {
public:
    bool load(const std::string& filename, std::string* error_message = nullptr);
    bool isLoaded() const noexcept { return loaded_; }

    std::array<float, 7> predictLogits(const ConnectFourState& state) const;
    int selectAction(const ConnectFourState& state) const;

private:
    bool loaded_ = false;

    // Conv: 2 -> 24 -> 48, kernel=3, padding=1
    std::vector<float> conv1_w_, conv1_b_;
    std::vector<float> conv2_w_, conv2_b_;

    // FC: 48*6*7 -> 96 -> 48 -> 7
    // Dropout은 추론 시 비활성화되므로 C++ 코드에 필요 없다.
    std::vector<float> fc1_w_, fc1_b_;
    std::vector<float> fc2_w_, fc2_b_;
    std::vector<float> fc3_w_, fc3_b_;
};
