//미니맥스 완성 후 머지 및 푸쉬 완료, 몬테카를로 만들차례
#include "minimax.hpp"
#include "ConnectFourState.hpp"
#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <iostream>
#include <chrono>
#include<random>
using State = ConnectFourState;
std::random_device rnd;
std::mt19937 mt_for_action(rnd());
static constexpr double INF = 100000000;

int randomAction(const State& state)
{
	auto legal_actions = state.legalActions();
	return legal_actions[mt_for_action() % (legal_actions.size())];
}
double playout(State state)
{
	// 게임 끝났으면, "현재 시점의 플레이어" 기준으로 평가
	if (state.isDone())
	{
		switch (state.getWinningStatus())
		{
		case WinningStatus::LOSE:
			// 현재 플레이어 입장: LOSE = 내가 이긴 상태
			return 0.0;
		case WinningStatus::WIN:
			// 현재 플레이어 입장: WIN  = 내가 진 상태
			return 1.0;
		case WinningStatus::DRAW:
			return 0.5;
		default:
			return 0.5;
		}
	}

	// 아직 안 끝났으면, 랜덤으로 한 수 두고
	state.advance(randomAction(state));
	// 턴이 바뀌었으니까, 값도 뒤집어 주기
	return 1.0 - playout(state);
}
int MontecarloAction(const State& state, int playout_num, int time_limit_ms)
{
	TimeKeeper tk(time_limit_ms);

	auto start = std::chrono::high_resolution_clock::now();
	auto legal_actions = state.legalActions();
	auto values = std::vector<double>(legal_actions.size());
	auto cnts = std::vector<double>(legal_actions.size());
	int playnum = 0;
	for (int cnt = 0; cnt < playout_num; cnt++)
	{
		if (tk.isTimeOver()) break;
		int index = cnt % legal_actions.size();
		State next_state = state;
		next_state.advance(legal_actions[index]);
		values[index] += 1.0 - playout(next_state);
		++cnts[index];
		playnum = cnts[index];

	}
	int best_action_index = -1;
	double best_score = -INF;
	for (int index = 0; index < legal_actions.size(); index++)
	{
		if (cnts[index] == 0.0)
		{
			continue;
		}

		double value_mean = values[index] / cnts[index];
		if (value_mean > best_score)
		{
			best_score = value_mean;
			best_action_index = index;
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration<double, std::milli>(end - start).count();
	//std::cout <<"playout_num: " << playnum << "\n";
	std::cout << 1;
	return legal_actions[best_action_index];
}
#include "Monte_Carlo.hpp"
#include <vector>
#include <cmath>
#include <memory>
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>

using State = ConnectFourState;

static std::mt19937 mt_for_action(std::random_device{}());

class TimeKeeper {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    int time_limit_ms_;

public:
    TimeKeeper(const int time_limit_ms)
        : start_time_(std::chrono::high_resolution_clock::now()),
        time_limit_ms_(time_limit_ms) {
    }

    bool isTimeOver() const {
        auto diff = std::chrono::high_resolution_clock::now() - start_time_;
        return std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() >= time_limit_ms_;
    }
};

int randomAction(const State& state) {
    auto legal_actions = state.legalActions();
    std::uniform_int_distribution<int> dist(0, (int)legal_actions.size() - 1);
    return legal_actions[dist(mt_for_action)];
}

// rollout 결과를 "현재 state에서 둘 차례인 플레이어" 기준으로 반환
// WIN  ->  1.0
// DRAW ->  0.0
// LOSE -> -1.0
double playout(State state) {
    if (state.isDone()) {
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

    while (!state.isDone()) {
        int action = randomAction(state);
        state.advance(action);
    }

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
            double value = playout(state_);
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

    std::cout << "MCTS playouts: " << count << "\n";
    return best_action;
}