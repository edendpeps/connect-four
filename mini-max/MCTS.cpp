#include "Monte_Carlo.hpp"
#include "MCTS.h"
#include "ConnectFourState.hpp"
#include <vector>
#include <cmath>
#include <memory>
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>

using State = ConnectFourState;

// rollout 결과를 현재 state에서 둘 차례인 플레이어 기준으로 반환
// WIN  ->  1.0
// DRAW ->  0.0
// LOSE -> -1.0
static double mctsplayout(State state) {
    if (state.isDone())
    {
        switch (state.getWinningStatus())
        {
        case WinningStatus::LOSE:
            return -1.0;
        case WinningStatus::WIN:
            return 1.0;
        case WinningStatus::DRAW:
            return 0.0;
        default:
            return 0.0;
        }
    }

    state.advance(randomAction(state));

    // advance 후 턴이 바뀌므로 부호 반전
    return -mctsplayout(state);
}

class Node {
public:
    State state_;
    double w_;
    int n_;
    int action_;
    std::vector<std::unique_ptr<Node>> child_nodes_;

    Node(const State& state, int action = -1)
        : state_(state), w_(0.0), n_(0), action_(action) {
    }

    bool isExpanded() const {
        return !child_nodes_.empty();
    }

    void expand() {
        auto legal_actions = state_.legalActions();
        child_nodes_.reserve(legal_actions.size());

        for (int action : legal_actions) {
            State next_state = state_;
            next_state.advance(action);
            child_nodes_.push_back(std::make_unique<Node>(next_state, action));
        }
    }

    Node* nextChildNode() {
        constexpr double C = 1.41421356237;

        for (auto& child : child_nodes_) {
            if (child->n_ == 0) return child.get();
        }

        double best_value = -1e18;
        Node* best_node = nullptr;

        for (auto& child : child_nodes_) {
            double exploitation = -(child->w_ / child->n_);
            double exploration = C * std::sqrt(std::log((double)(n_ + 1)) / child->n_);
            double uct_value = exploitation + exploration;

            if (uct_value > best_value) {
                best_value = uct_value;
                best_node = child.get();
            }
        }

        return best_node;
    }

    double evaluate() {
        if (state_.isDone()) {
            double value = 0.0;
            switch (state_.getWinningStatus()) {
            case WinningStatus::WIN:
                value = 1.0;
                break;
            case WinningStatus::LOSE:
                value = -1.0;
                break;
            case WinningStatus::DRAW:
                value = 0.0;
                break;
            default:
                value = 0.0;
                break;
            }

            n_++;
            w_ += value;
            return value;
        }

        if (!isExpanded()) {
            double value = mctsplayout(state_);
            n_++;
            w_ += value;

            if (n_ == 1) {
                expand();
            }

            return value;
        }
        else {
            Node* child = nextChildNode();
            double value = -child->evaluate();
            n_++;
            w_ += value;
            return value;
        }
    }
};

int MCTSAction(const State& state, int playout_number, int time_limit_ms) {
    TimeKeeper tk(time_limit_ms);
    Node root_node(state);
    root_node.expand();

    int count = 0;
    while (!tk.isTimeOver() && count < playout_number) {
        root_node.evaluate();
        count++;
    }

    auto legal_actions = state.legalActions();
    if (root_node.child_nodes_.empty()) {
        return legal_actions.empty() ? -1 : legal_actions.front();
    }

    int best_action = root_node.child_nodes_[0]->action_;
    int best_n = root_node.child_nodes_[0]->n_;

    for (auto& child : root_node.child_nodes_) {
        if (child->n_ > best_n) {
            best_n = child->n_;
            best_action = child->action_;
        }
    }

    return best_action;
}