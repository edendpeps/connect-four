#include "FairMCTS.hpp"

#include "MLPPolicy.hpp"
#include "CNNPolicy.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

namespace {

using State = ConnectFourState;
using Clock = std::chrono::steady_clock;

constexpr std::array<double, 7> RANK_WEIGHTS = {
    1.00, 0.75, 0.55, 0.40, 0.30, 0.22, 0.16
};

struct RootPolicyData {
    bool enabled = false;
    std::array<float, 7> priors{};
    int evaluations = 0;
    std::int64_t inference_ns = 0;
};

double terminalValue(const State& state)
{
    switch (state.getWinningStatus()) {
    case WinningStatus::WIN:  return 1.0;
    case WinningStatus::LOSE: return -1.0;
    case WinningStatus::DRAW: return 0.0;
    default:                   return 0.0;
    }
}

int randomLegalAction(const State& state, std::mt19937& rng)
{
    const auto legal = state.legalActions();
    if (legal.empty()) return -1;

    std::uniform_int_distribution<int> dist(
        0,
        static_cast<int>(legal.size()) - 1
    );
    return legal[static_cast<std::size_t>(dist(rng))];
}

double randomPlayout(State state, std::mt19937& rng)
{
    double sign = 1.0;

    while (!state.isDone()) {
        const int action = randomLegalAction(state, rng);
        if (action < 0) return 0.0;

        state.advance(action);
        sign = -sign;
    }

    return sign * terminalValue(state);
}

std::array<float, 7> buildRankPrior(
    const State& state,
    const std::array<float, 7>& logits,
    double uniform_mix
)
{
    std::array<float, 7> priors{};
    auto legal = state.legalActions();
    if (legal.empty()) return priors;

    std::stable_sort(
        legal.begin(),
        legal.end(),
        [&](int a, int b) {
            return logits[static_cast<std::size_t>(a)]
                > logits[static_cast<std::size_t>(b)];
        }
    );

    double rank_sum = 0.0;
    for (std::size_t rank = 0; rank < legal.size(); ++rank) {
        const double weight = RANK_WEIGHTS[rank];
        priors[static_cast<std::size_t>(legal[rank])] =
            static_cast<float>(weight);
        rank_sum += weight;
    }

    uniform_mix = std::max(0.0, std::min(1.0, uniform_mix));
    const double uniform = 1.0 / static_cast<double>(legal.size());

    for (int action : legal) {
        const double rank_prior =
            priors[static_cast<std::size_t>(action)] / rank_sum;

        priors[static_cast<std::size_t>(action)] =
            static_cast<float>(
                (1.0 - uniform_mix) * rank_prior
                + uniform_mix * uniform
            );
    }

    return priors;
}

class Node {
public:
    Node(
        const State& state,
        int action = -1,
        float prior = 0.0f,
        bool use_root_policy = false
    )
        : state_(state),
          action_(action),
          prior_(prior),
          use_root_policy_(use_root_policy)
    {
    }

    bool isExpanded() const
    {
        return !children_.empty();
    }

    void expand(const std::array<float, 7>* priors = nullptr)
    {
        if (state_.isDone() || isExpanded()) return;

        const auto legal = state_.legalActions();
        children_.reserve(legal.size());

        for (int action : legal) {
            State next = state_;
            next.advance(action);

            const float prior = priors == nullptr
                ? 0.0f
                : (*priors)[static_cast<std::size_t>(action)];

            children_.push_back(
                std::make_unique<Node>(next, action, prior, false)
            );
        }
    }

    Node* nextChild(std::mt19937& rng, const FairMCTSConfig& config)
    {
        std::vector<Node*> unvisited;
        unvisited.reserve(children_.size());

        for (auto& child : children_) {
            if (child->visits_ == 0) {
                unvisited.push_back(child.get());
            }
        }

        if (!unvisited.empty()) {
            if (use_root_policy_) {
                float best_prior = -std::numeric_limits<float>::infinity();
                std::vector<Node*> best;

                for (Node* child : unvisited) {
                    if (child->prior_ > best_prior + 1e-12f) {
                        best_prior = child->prior_;
                        best.clear();
                        best.push_back(child);
                    }
                    else if (std::fabs(child->prior_ - best_prior) <= 1e-12f) {
                        best.push_back(child);
                    }
                }

                std::uniform_int_distribution<int> dist(
                    0,
                    static_cast<int>(best.size()) - 1
                );
                return best[static_cast<std::size_t>(dist(rng))];
            }

            // 일반 MCTS와 하위 노드는 방문하지 않은 수 중 무작위 선택.
            std::uniform_int_distribution<int> dist(
                0,
                static_cast<int>(unvisited.size()) - 1
            );
            return unvisited[static_cast<std::size_t>(dist(rng))];
        }

        const double log_parent =
            std::log(std::max(1.0, static_cast<double>(visits_)));
        const double uniform_prior =
            1.0 / static_cast<double>(children_.size());

        double best_value = -std::numeric_limits<double>::infinity();
        std::vector<Node*> best;

        for (auto& child_ptr : children_) {
            Node* child = child_ptr.get();

            const double exploitation =
                -(child->value_sum_ / static_cast<double>(child->visits_));

            const double exploration =
                config.uct_c
                * std::sqrt(
                    log_parent / static_cast<double>(child->visits_)
                );

            const double positive_prior = use_root_policy_
                ? std::max(
                    0.0,
                    static_cast<double>(child->prior_) - uniform_prior
                )
                : 0.0;

            const double policy_bias = use_root_policy_
                ? config.root_policy_weight
                    * positive_prior
                    / std::sqrt(
                        1.0 + static_cast<double>(child->visits_)
                    )
                : 0.0;

            const double value =
                exploitation + exploration + policy_bias;

            if (value > best_value + 1e-12) {
                best_value = value;
                best.clear();
                best.push_back(child);
            }
            else if (std::fabs(value - best_value) <= 1e-12) {
                best.push_back(child);
            }
        }

        if (best.empty()) return nullptr;

        std::uniform_int_distribution<int> dist(
            0,
            static_cast<int>(best.size()) - 1
        );
        return best[static_cast<std::size_t>(dist(rng))];
    }

    double evaluate(std::mt19937& rng, const FairMCTSConfig& config)
    {
        if (state_.isDone()) {
            const double value = terminalValue(state_);
            ++visits_;
            value_sum_ += value;
            return value;
        }

        if (!isExpanded()) {
            const double value = randomPlayout(state_, rng);
            ++visits_;
            value_sum_ += value;

            if (visits_ == 1) expand();
            return value;
        }

        Node* child = nextChild(rng, config);
        if (child == nullptr) {
            ++visits_;
            return 0.0;
        }

        const double value = -child->evaluate(rng, config);
        ++visits_;
        value_sum_ += value;
        return value;
    }

    int action() const { return action_; }
    int visits() const { return visits_; }
    float prior() const { return prior_; }

    const std::vector<std::unique_ptr<Node>>& children() const
    {
        return children_;
    }

private:
    State state_;
    double value_sum_ = 0.0;
    int visits_ = 0;
    int action_ = -1;
    float prior_ = 0.0f;
    bool use_root_policy_ = false;
    std::vector<std::unique_ptr<Node>> children_;
};

template <typename Policy>
RootPolicyData evaluatePolicy(
    const State& state,
    const Policy& policy,
    const FairMCTSConfig& config
)
{
    RootPolicyData data;
    data.enabled = true;

    const auto begin = Clock::now();
    const auto logits = policy.predictLogits(state);
    const auto end = Clock::now();

    data.priors = buildRankPrior(state, logits, config.uniform_mix);
    data.evaluations = 1;
    data.inference_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
            .count();

    return data;
}

FairMCTSResult runSearch(
    const State& state,
    const FairMCTSConfig& config,
    const RootPolicyData& policy_data,
    std::mt19937& rng,
    TimeKeeper& time_keeper
)
{
    FairMCTSResult result;
    const auto legal = state.legalActions();
    if (legal.empty()) return result;

    Node root(state, -1, 0.0f, policy_data.enabled);
    root.expand(policy_data.enabled ? &policy_data.priors : nullptr);

    const int simulation_limit = std::max(1, config.simulations);
    int simulations = 0;

    while (true) {
        if (config.budget_mode
            == FairSearchBudgetMode::FixedSimulations) {
            if (simulations >= simulation_limit) break;
        }
        else {
            // 정책망 추론 시간까지 제한시간에 포함한다.
            if (simulations > 0 && time_keeper.isTimeOver()) break;
        }

        root.evaluate(rng, config);
        ++simulations;
    }

    int best_visits = -1;
    float best_prior = -std::numeric_limits<float>::infinity();
    std::vector<const Node*> best;

    for (const auto& child_ptr : root.children()) {
        const Node* child = child_ptr.get();

        const bool better_visits = child->visits() > best_visits;
        const bool equal_visits = child->visits() == best_visits;
        const bool better_prior = policy_data.enabled
            && child->prior() > best_prior + 1e-12f;
        const bool equal_prior = !policy_data.enabled
            || std::fabs(child->prior() - best_prior) <= 1e-12f;

        if (better_visits || (equal_visits && better_prior)) {
            best_visits = child->visits();
            best_prior = child->prior();
            best.clear();
            best.push_back(child);
        }
        else if (equal_visits && equal_prior) {
            best.push_back(child);
        }
    }

    if (best.empty()) {
        result.action = legal.front();
    }
    else {
        std::uniform_int_distribution<int> dist(
            0,
            static_cast<int>(best.size()) - 1
        );
        result.action = best[static_cast<std::size_t>(dist(rng))]->action();
    }

    result.simulations = simulations;
    result.policy_evaluations = policy_data.evaluations;
    result.policy_inference_ns = policy_data.inference_ns;
    return result;
}

} // namespace

FairMCTSResult PlainMCTSAction(
    const ConnectFourState& state,
    const FairMCTSConfig& config,
    std::mt19937& rng
)
{
    TimeKeeper time_keeper(std::max(1, config.time_limit_ms));
    RootPolicyData none;
    return runSearch(state, config, none, rng, time_keeper);
}

FairMCTSResult MLPRootMCTSAction(
    const ConnectFourState& state,
    const MLPPolicy& policy,
    const FairMCTSConfig& config,
    std::mt19937& rng
)
{
    TimeKeeper time_keeper(std::max(1, config.time_limit_ms));
    const auto data = evaluatePolicy(state, policy, config);
    return runSearch(state, config, data, rng, time_keeper);
}

FairMCTSResult CNNRootMCTSAction(
    const ConnectFourState& state,
    const CNNPolicy& policy,
    const FairMCTSConfig& config,
    std::mt19937& rng
)
{
    TimeKeeper time_keeper(std::max(1, config.time_limit_ms));
    const auto data = evaluatePolicy(state, policy, config);
    return runSearch(state, config, data, rng, time_keeper);
}
