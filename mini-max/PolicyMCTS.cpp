#include "PolicyMCTS.hpp"

#include "MLPPolicy.hpp"
#include "CNNPolicy.hpp"
#include "Monte_Carlo.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace {

    using State = ConnectFourState;
    using Clock = std::chrono::steady_clock;

    // 정책망은 루트에서 한 번만 사용한다.
    // 원래 UCT 탐색은 그대로 유지하고, 정책은 초반 탐색 순서와
    // 루트의 작은 progressive bias에만 사용한다.
    constexpr double UCT_C = 1.41421356237;

    // one-hot best_action으로 학습한 정책망은 확률 보정보다
    // '수의 순위' 정보가 더 믿을 만하므로 rank prior를 사용한다.
    constexpr double ROOT_POLICY_WEIGHT = 0.80;
    constexpr double UNIFORM_MIX = 0.40;

    constexpr std::array<double, 7> RANK_WEIGHTS = {
        1.00, 0.75, 0.55, 0.40, 0.30, 0.22, 0.16
    };

    double terminalValue(const State& state)
    {
        switch (state.getWinningStatus()) {
        case WinningStatus::WIN:
            return 1.0;
        case WinningStatus::LOSE:
            return -1.0;
        case WinningStatus::DRAW:
            return 0.0;
        default:
            return 0.0;
        }
    }

    // 원래 MCTS.cpp와 같은 관점 규칙을 사용한다.
    double mctsPlayout(State state)
    {
        if (state.isDone()) {
            return terminalValue(state);
        }

        state.advance(randomAction(state));
        return -mctsPlayout(state);
    }

    std::array<float, 7> buildRankPrior(
        const State& state,
        const std::array<float, 7>& logits
    )
    {
        std::array<float, 7> priors{};
        auto legal = state.legalActions();

        if (legal.empty()) {
            return priors;
        }

        // softmax 값 자체를 확률처럼 믿지 않고,
        // logit이 큰 순서만 사용한다.
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

        const double uniform =
            1.0 / static_cast<double>(legal.size());

        // 정책 60% + 균등분포 40%.
        // 낮은 순위의 합법 수도 완전히 버리지 않는다.
        for (int action : legal) {
            const double rank_prior =
                priors[static_cast<std::size_t>(action)] / rank_sum;

            priors[static_cast<std::size_t>(action)] =
                static_cast<float>(
                    (1.0 - UNIFORM_MIX) * rank_prior
                    + UNIFORM_MIX * uniform
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
            return !child_nodes_.empty();
        }

        // priors가 전달되는 것은 루트 확장 한 번뿐이다.
        void expand(const std::array<float, 7>* priors = nullptr)
        {
            if (state_.isDone() || isExpanded()) {
                return;
            }

            const auto legal_actions = state_.legalActions();
            child_nodes_.reserve(legal_actions.size());

            for (int action : legal_actions) {
                State next_state = state_;
                next_state.advance(action);

                const float prior =
                    priors == nullptr
                    ? 0.0f
                    : (*priors)[static_cast<std::size_t>(action)];

                child_nodes_.push_back(
                    std::make_unique<Node>(
                        next_state,
                        action,
                        prior,
                        false
                    )
                );
            }
        }

        Node* nextChildNode()
        {
            // 원래 MCTS처럼 방문하지 않은 수는 반드시 먼저 탐색한다.
            // 루트에서만 정책 확률이 높은 수부터 확인한다.
            Node* unvisited = nullptr;

            for (auto& child : child_nodes_) {
                if (child->n_ != 0) {
                    continue;
                }

                if (!use_root_policy_) {
                    return child.get();
                }

                if (
                    unvisited == nullptr
                    || child->prior_ > unvisited->prior_
                    ) {
                    unvisited = child.get();
                }
            }

            if (unvisited != nullptr) {
                return unvisited;
            }

            double best_value =
                -std::numeric_limits<double>::infinity();

            Node* best_node = nullptr;

            for (auto& child : child_nodes_) {
                const double exploitation =
                    -(child->w_ / static_cast<double>(child->n_));

                const double exploration =
                    UCT_C * std::sqrt(
                        std::log(static_cast<double>(n_))
                        / static_cast<double>(child->n_)
                    );

                // 균등분포보다 높은 prior에만 보너스를 준다.
                // 낮은 순위의 수를 직접 벌점 처리하지는 않는다.
                const double uniform_prior =
                    1.0 / static_cast<double>(child_nodes_.size());

                const double positive_prior =
                    std::max(
                        0.0,
                        static_cast<double>(child->prior_)
                        - uniform_prior
                    );

                const double policy_bias =
                    use_root_policy_
                    ? ROOT_POLICY_WEIGHT
                    * positive_prior
                    / std::sqrt(
                        1.0 + static_cast<double>(child->n_)
                    )
                    : 0.0;

                const double value =
                    exploitation + exploration + policy_bias;

                if (value > best_value) {
                    best_value = value;
                    best_node = child.get();
                }
            }

            return best_node;
        }

        double evaluate()
        {
            if (state_.isDone()) {
                const double value = terminalValue(state_);
                n_++;
                w_ += value;
                return value;
            }

            if (!isExpanded()) {
                const double value = mctsPlayout(state_);

                n_++;
                w_ += value;

                if (n_ == 1) {
                    expand();
                }

                return value;
            }

            Node* child = nextChildNode();

            if (child == nullptr) {
                n_++;
                return 0.0;
            }

            const double value = -child->evaluate();

            n_++;
            w_ += value;

            return value;
        }

        int action() const
        {
            return action_;
        }

        int visits() const
        {
            return n_;
        }

        float prior() const
        {
            return prior_;
        }

        const std::vector<std::unique_ptr<Node>>& children() const
        {
            return child_nodes_;
        }

    private:
        State state_;
        double w_ = 0.0;
        int n_ = 0;
        int action_ = -1;
        float prior_ = 0.0f;

        // true인 노드는 루트 하나뿐이다.
        bool use_root_policy_ = false;

        std::vector<std::unique_ptr<Node>> child_nodes_;
    };

    template <typename Policy>
    PolicyMCTSResult runRootPolicyMCTS(
        const State& state,
        const Policy& policy,
        const PolicyMCTSConfig& config
    )
    {
        PolicyMCTSResult result;

        const auto legal = state.legalActions();

        if (legal.empty()) {
            return result;
        }

        const auto search_begin = Clock::now();

        // 정책망은 한 수마다 루트에서 딱 한 번만 실행한다.
        const auto policy_begin = Clock::now();
        const auto logits = policy.predictLogits(state);
        const auto policy_end = Clock::now();

        const std::int64_t policy_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                policy_end - policy_begin
            ).count();

        const auto priors = buildRankPrior(state, logits);

        Node root(state, -1, 0.0f, true);
        root.expand(&priors);

        const auto deadline =
            search_begin
            + std::chrono::milliseconds(
                std::max(1, config.time_limit_ms)
            );

        int simulations = 0;

        while (true) {
            if (
                config.budget_mode
                == SearchBudgetMode::FixedSimulations
                ) {
                if (
                    simulations
                    >= std::max(1, config.simulations)
                    ) {
                    break;
                }
            }
            else {
                if (
                    simulations > 0
                    && Clock::now() >= deadline
                    ) {
                    break;
                }
            }

            root.evaluate();
            simulations++;
        }

        const Node* best = nullptr;

        for (const auto& child : root.children()) {
            if (
                best == nullptr
                || child->visits() > best->visits()
                || (
                    child->visits() == best->visits()
                    && child->prior() > best->prior()
                    )
                ) {
                best = child.get();
            }
        }

        result.action =
            best == nullptr
            ? legal.front()
            : best->action();

        result.simulations = simulations;
        result.policy_evaluations = 1;
        result.policy_inference_ns = policy_ns;

        return result;
    }

} // namespace

PolicyMCTSResult MLPMCTSAction(
    const ConnectFourState& state,
    const MLPPolicy& policy,
    const PolicyMCTSConfig& config,
    std::mt19937& rng
)
{
    (void)rng;

    return runRootPolicyMCTS(
        state,
        policy,
        config
    );
}

PolicyMCTSResult CNNMCTSAction(
    const ConnectFourState& state,
    const CNNPolicy& policy,
    const PolicyMCTSConfig& config,
    std::mt19937& rng
)
{
    (void)rng;

    return runRootPolicyMCTS(
        state,
        policy,
        config
    );
}