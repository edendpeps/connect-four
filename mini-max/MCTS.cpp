#include "Monte_Carlo.hpp"
#include <vector>
#include <cmath>
#include <memory>
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>

using State = ConnectFourState;

// rollout 결과를 "현재 state에서 둘 차례인 플레이어" 기준으로 반환
// WIN  ->  1.0
// DRAW ->  0.0
// LOSE -> -1.0
static double mctsplayout(State state) {
    // 게임 끝났으면, "현재 시점의 플레이어" 기준으로 평가
    if (state.isDone())
    {
        switch (state.getWinningStatus())
        {
        case WinningStatus::LOSE:
            // 현재 플레이어 입장: LOSE = 내가 이긴 상태
            return -1.0;
        case WinningStatus::WIN:
            // 현재 플레이어 입장: WIN  = 내가 진 상태
            return 1.0;
        case WinningStatus::DRAW:
            return 0.0;
        default:
            return 0.0;
        }
    }

    // 아직 안 끝났으면, 랜덤으로 한 수 두고
    state.advance(randomAction(state));
    // 턴이 바뀌었으니까, 값도 뒤집어 주기
    return 1.0 - mctsplayout(state);
}

class Node {
public:
    State state_;
    double w_;   // 누적 가치합
    int n_;      // 방문 수
    int action_; // 부모에서 여기로 오게 한 수
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

    // UCT 값 최대 자식 선택
    Node* nextChildNode() {
        constexpr double C = 1.41421356237; // sqrt(2)

        // 아직 방문 안 한 자식 우선
        for (auto& child : child_nodes_) {
            if (child->n_ == 0) return child.get();
        }

        double best_value = -1e18;
        Node* best_node = nullptr;

        for (auto& child : child_nodes_) {
            // child는 "상대 차례 state"라서 부호 반전해서 해석
            double exploitation = -(child->w_ / child->n_);
            double exploration = C * std::sqrt(std::log((double)n_) / child->n_);
            double uct_value = exploitation + exploration;

            if (uct_value > best_value) {
                best_value = uct_value;
                best_node = child.get();
            }
        }

        return best_node;
    }

    // 현재 node에서 본 가치 반환
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

            // 한 번 이상 방문되면 확장
            if (n_ == 1) {
                expand();
            }

            return value;
        }
        else {
            Node* child = nextChildNode();
            double value = -child->evaluate(); // 턴 교대 반영
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

   //std::cout << "MCTS playouts: " << count << "\n";
    return best_action;
}