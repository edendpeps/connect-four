#include "CNNPolicy.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace {
constexpr char kMagic[8] = { 'C', '4', 'P', 'O', 'L', 'I', 'C', 'Y' };
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kModelTypeCNN = 2;

bool readExact(std::ifstream& in, void* dst, std::size_t bytes) {
    in.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(in);
}

bool readTensor(
    std::ifstream& in,
    std::vector<float>& dst,
    std::uint64_t expected_count,
    std::string& error
) {
    std::uint64_t count = 0;
    if (!readExact(in, &count, sizeof(count))) {
        error = "가중치 파일에서 tensor 크기를 읽지 못했습니다.";
        return false;
    }
    if (count != expected_count) {
        error = "CNN tensor 크기가 모델 구조와 다릅니다. expected="
            + std::to_string(expected_count) + ", actual=" + std::to_string(count);
        return false;
    }

    dst.resize(static_cast<std::size_t>(count));
    if (!readExact(in, dst.data(), dst.size() * sizeof(float))) {
        error = "가중치 파일에서 tensor 데이터를 끝까지 읽지 못했습니다.";
        return false;
    }
    return true;
}

std::vector<float> conv3x3Padding1(
    const std::vector<float>& input,
    int in_channels,
    int out_channels,
    const std::vector<float>& weight,
    const std::vector<float>& bias
) {
    std::vector<float> output(
        static_cast<std::size_t>(out_channels) * H * W,
        0.0f
    );

    for (int oc = 0; oc < out_channels; ++oc) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float sum = bias[static_cast<std::size_t>(oc)];

                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int ky = 0; ky < 3; ++ky) {
                        for (int kx = 0; kx < 3; ++kx) {
                            const int iy = y + ky - 1;
                            const int ix = x + kx - 1;
                            if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;

                            const std::size_t input_index =
                                (static_cast<std::size_t>(ic) * H + iy) * W + ix;
                            const std::size_t weight_index =
                                (((static_cast<std::size_t>(oc) * in_channels + ic) * 3 + ky) * 3 + kx);

                            sum += input[input_index] * weight[weight_index];
                        }
                    }
                }

                // ReLU
                if (sum < 0.0f) sum = 0.0f;
                const std::size_t output_index =
                    (static_cast<std::size_t>(oc) * H + y) * W + x;
                output[output_index] = sum;
            }
        }
    }

    return output;
}

std::vector<float> dense(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const std::vector<float>& bias,
    int out_features,
    int in_features,
    bool use_relu
) {
    std::vector<float> output(static_cast<std::size_t>(out_features), 0.0f);

    for (int o = 0; o < out_features; ++o) {
        float sum = bias[static_cast<std::size_t>(o)];
        const std::size_t row = static_cast<std::size_t>(o) * in_features;

        for (int i = 0; i < in_features; ++i) {
            sum += weight[row + static_cast<std::size_t>(i)]
                * input[static_cast<std::size_t>(i)];
        }

        if (use_relu && sum < 0.0f) sum = 0.0f;
        output[static_cast<std::size_t>(o)] = sum;
    }

    return output;
}

std::vector<float> encodeBoardTwoChannels(const ConnectFourState& state) {
    std::vector<float> input(2ULL * H * W, 0.0f);

    const int(*my)[W] = state.getMyBoard();
    const int(*enemy)[W] = state.getEnemyBoard();

    // PyTorch NCHW 순서: channel -> y -> x
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const std::size_t pos = static_cast<std::size_t>(y) * W + x;
            if (my[y][x] == 1) input[pos] = 1.0f;
            if (enemy[y][x] == 1) input[static_cast<std::size_t>(H * W) + pos] = 1.0f;
        }
    }

    return input;
}

int bestLegalAction(
    const ConnectFourState& state,
    const std::array<float, 7>& logits
) {
    const auto legal = state.legalActions();
    if (legal.empty()) return -1;

    constexpr int center_order[7] = { 3, 2, 4, 1, 5, 0, 6 };
    bool legal_mask[7] = {};
    for (int action : legal) legal_mask[action] = true;

    int best_action = legal.front();
    float best_value = -std::numeric_limits<float>::infinity();

    for (int action : center_order) {
        if (!legal_mask[action]) continue;
        if (logits[static_cast<std::size_t>(action)] > best_value) {
            best_value = logits[static_cast<std::size_t>(action)];
            best_action = action;
        }
    }

    return best_action;
}
} // namespace

bool CNNPolicy::load(const std::string& filename, std::string* error_message) {
    loaded_ = false;
    std::string error;

    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        error = "CNN 가중치 파일을 열 수 없습니다: " + filename;
        if (error_message) *error_message = error;
        return false;
    }

    char magic[8] = {};
    std::uint32_t version = 0;
    std::uint32_t model_type = 0;
    std::uint32_t tensor_count = 0;

    if (!readExact(in, magic, sizeof(magic))
        || !readExact(in, &version, sizeof(version))
        || !readExact(in, &model_type, sizeof(model_type))
        || !readExact(in, &tensor_count, sizeof(tensor_count))) {
        error = "CNN 가중치 파일 헤더가 손상되었습니다.";
    }
    else if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        error = "CNN 가중치 파일 magic 값이 올바르지 않습니다.";
    }
    else if (version != kVersion || model_type != kModelTypeCNN || tensor_count != 10) {
        error = "CNN 가중치 파일의 버전 또는 모델 종류가 맞지 않습니다.";
    }
    else if (!readTensor(in, conv1_w_, 24ULL * 2ULL * 3ULL * 3ULL, error)
        || !readTensor(in, conv1_b_, 24ULL, error)
        || !readTensor(in, conv2_w_, 48ULL * 24ULL * 3ULL * 3ULL, error)
        || !readTensor(in, conv2_b_, 48ULL, error)
        || !readTensor(in, fc1_w_, 96ULL * (48ULL * 6ULL * 7ULL), error)
        || !readTensor(in, fc1_b_, 96ULL, error)
        || !readTensor(in, fc2_w_, 48ULL * 96ULL, error)
        || !readTensor(in, fc2_b_, 48ULL, error)
        || !readTensor(in, fc3_w_, 7ULL * 48ULL, error)
        || !readTensor(in, fc3_b_, 7ULL, error)) {
        // error는 readTensor에서 작성됨.
    }
    else {
        loaded_ = true;
    }

    if (!loaded_ && error_message) *error_message = error;
    return loaded_;
}

std::array<float, 7> CNNPolicy::predictLogits(const ConnectFourState& state) const {
    if (!loaded_) throw std::runtime_error("CNNPolicy가 로드되지 않았습니다.");

    const auto input = encodeBoardTwoChannels(state);
    const auto c1 = conv3x3Padding1(input, 2, 24, conv1_w_, conv1_b_);
    const auto c2 = conv3x3Padding1(c1, 24, 48, conv2_w_, conv2_b_);

    // c2가 이미 PyTorch Flatten과 같은 C-major contiguous 순서이다.
    const auto h1 = dense(c2, fc1_w_, fc1_b_, 96, 48 * 6 * 7, true);
    const auto h2 = dense(h1, fc2_w_, fc2_b_, 48, 96, true);
    const auto out = dense(h2, fc3_w_, fc3_b_, 7, 48, false);

    std::array<float, 7> logits{};
    std::copy(out.begin(), out.end(), logits.begin());
    return logits;
}

int CNNPolicy::selectAction(const ConnectFourState& state) const {
    return bestLegalAction(state, predictLogits(state));
}
