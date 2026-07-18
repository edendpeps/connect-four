#include "MLPPolicy.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace {
constexpr char kMagic[8] = { 'C', '4', 'P', 'O', 'L', 'I', 'C', 'Y' };
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kModelTypeMLP = 1;

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
        error = "MLP tensor 크기가 모델 구조와 다릅니다. expected="
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

std::vector<float> encodeBoard(const ConnectFourState& state) {
    std::vector<float> input;
    input.reserve(42);

    const int(*my)[W] = state.getMyBoard();
    const int(*enemy)[W] = state.getEnemyBoard();

    // Python Dataset과 같은 순서: y=0..5, x=0..6
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float value = 0.0f;
            if (my[y][x] == 1) value = 1.0f;
            else if (enemy[y][x] == 1) value = -1.0f;
            input.push_back(value);
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

    // 완전히 같은 logit이면 중앙 쪽을 우선한다.
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

bool MLPPolicy::load(const std::string& filename, std::string* error_message) {
    loaded_ = false;
    std::string error;

    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        error = "MLP 가중치 파일을 열 수 없습니다: " + filename;
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
        error = "MLP 가중치 파일 헤더가 손상되었습니다.";
    }
    else if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        error = "MLP 가중치 파일 magic 값이 올바르지 않습니다.";
    }
    else if (version != kVersion || model_type != kModelTypeMLP || tensor_count != 8) {
        error = "MLP 가중치 파일의 버전 또는 모델 종류가 맞지 않습니다.";
    }
    else if (!readTensor(in, w1_, 128ULL * 42ULL, error)
        || !readTensor(in, b1_, 128ULL, error)
        || !readTensor(in, w2_, 64ULL * 128ULL, error)
        || !readTensor(in, b2_, 64ULL, error)
        || !readTensor(in, w3_, 32ULL * 64ULL, error)
        || !readTensor(in, b3_, 32ULL, error)
        || !readTensor(in, w4_, 7ULL * 32ULL, error)
        || !readTensor(in, b4_, 7ULL, error)) {
        // error는 readTensor에서 작성됨.
    }
    else {
        loaded_ = true;
    }

    if (!loaded_ && error_message) *error_message = error;
    return loaded_;
}

std::array<float, 7> MLPPolicy::predictLogits(const ConnectFourState& state) const {
    if (!loaded_) throw std::runtime_error("MLPPolicy가 로드되지 않았습니다.");

    const auto x = encodeBoard(state);
    const auto h1 = dense(x, w1_, b1_, 128, 42, true);
    const auto h2 = dense(h1, w2_, b2_, 64, 128, true);
    const auto h3 = dense(h2, w3_, b3_, 32, 64, true);
    const auto out = dense(h3, w4_, b4_, 7, 32, false);

    std::array<float, 7> logits{};
    std::copy(out.begin(), out.end(), logits.begin());
    return logits;
}

int MLPPolicy::selectAction(const ConnectFourState& state) const {
    return bestLegalAction(state, predictLogits(state));
}
