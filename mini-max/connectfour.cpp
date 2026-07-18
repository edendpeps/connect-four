#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ConnectFourState.hpp"
#include "minimax.hpp"
#include "Monte_Carlo.hpp"
#include "MCTS.h"

#include "MLPPolicy.hpp"
#include "CNNPolicy.hpp"
#include "PolicyMCTS.hpp"

// ==================== 여기만 수정 ====================
const int GAME_LIMIT = 200;          // 대진 하나당 총 판수
const int TIME_LIMIT_MS = 100;       // 한 수당 제한시간
const int INF = 100000000;
const int RANDOM_SEED = 42;
// ====================================================

using State = ConnectFourState;

enum class AIType {
    AlphaBeta,
    MCTS,
    PureMC,
    MLPMCTS,
    CNNMCTS
};

MLPPolicy mlp_policy;
CNNPolicy cnn_policy;
PolicyMCTSConfig policy_mcts_config;
std::mt19937 policy_rng(RANDOM_SEED);

struct MatchStats {
    int a_win = 0;
    int b_win = 0;
    int draw = 0;

    long long a_time_ns = 0;
    long long b_time_ns = 0;
    long long a_moves = 0;
    long long b_moves = 0;
};

const char* aiName(AIType ai)
{
    switch (ai) {
    case AIType::AlphaBeta: return "AlphaBeta";
    case AIType::MCTS:      return "MCTS";
    case AIType::PureMC:    return "PureMC";
    case AIType::MLPMCTS:   return "MLP_MCTS";
    case AIType::CNNMCTS:   return "CNN_MCTS";
    default:                return "Unknown";
    }
}

int chooseAction(AIType ai, const State& state)
{
    switch (ai) {
    case AIType::AlphaBeta:
        return alphaBetaAction(state, INF, TIME_LIMIT_MS);

    case AIType::MCTS:
        return MCTSAction(state, INF, TIME_LIMIT_MS);

    case AIType::PureMC:
        return MontecarloAction(state, INF, TIME_LIMIT_MS);

    case AIType::MLPMCTS:
        return MLPMCTSAction(
            state,
            mlp_policy,
            policy_mcts_config,
            policy_rng
        ).action;

    case AIType::CNNMCTS:
        return CNNMCTSAction(
            state,
            cnn_policy,
            policy_mcts_config,
            policy_rng
        ).action;
    }

    return -1;
}

// A와 B가 빈 보드에서 한 판 대국한다.
// a_first가 true이면 A 선공, false이면 B 선공이다.
void playGame(
    AIType a,
    AIType b,
    bool a_first,
    MatchStats& stats
)
{
    State state;

    bool a_turn = a_first;
    bool last_move_by_a = false;

    while (!state.isDone()) {
        const AIType current_ai = a_turn ? a : b;

        const auto begin = std::chrono::steady_clock::now();
        const int action = chooseAction(current_ai, state);
        const auto end = std::chrono::steady_clock::now();

        const long long elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - begin
            ).count();

        if (a_turn) {
            stats.a_time_ns += elapsed;
            stats.a_moves++;
        }
        else {
            stats.b_time_ns += elapsed;
            stats.b_moves++;
        }

        const auto legal = state.legalActions();
        if (std::find(legal.begin(), legal.end(), action) == legal.end()) {
            throw std::runtime_error(
                std::string(aiName(current_ai))
                + "가 불가능한 수를 선택했습니다: "
                + std::to_string(action)
            );
        }

        state.advance(action);

        last_move_by_a = a_turn;
        a_turn = !a_turn;
    }

    if (state.getWinningStatus() == WinningStatus::DRAW) {
        stats.draw++;
    }
    else if (state.getWinningStatus() == WinningStatus::LOSE) {
        // advance() 후에는 다음 플레이어 관점이므로,
        // LOSE이면 방금 둔 플레이어가 승리한 것이다.
        if (last_move_by_a) stats.a_win++;
        else stats.b_win++;
    }
    else {
        // 현재 ConnectFourState 구조에서는 거의 나오지 않는 안전 처리
        if (last_move_by_a) stats.b_win++;
        else stats.a_win++;
    }
}

void runMatch(AIType a, AIType b)
{
    MatchStats stats;

    std::cout << "\n========== "
        << aiName(a) << " vs " << aiName(b)
        << " ==========\n";

    for (int game = 0; game < GAME_LIMIT; ++game) {
        // 매 판마다 선공 교환
        const bool a_first = (game % 2 == 0);
        playGame(a, b, a_first, stats);

        if ((game + 1) % 20 == 0 || game + 1 == GAME_LIMIT) {
            std::cout << "progress: "
                << game + 1 << " / " << GAME_LIMIT
                << " games\n";
        }
    }

    const int total = stats.a_win + stats.b_win + stats.draw;
    const int no_draw = stats.a_win + stats.b_win;

    std::cout << "Total games: " << total << '\n';

    std::cout << aiName(a) << " wins: "
        << stats.a_win << " ("
        << std::fixed << std::setprecision(2)
        << 100.0 * stats.a_win / total << "%)\n";

    std::cout << aiName(b) << " wins: "
        << stats.b_win << " ("
        << 100.0 * stats.b_win / total << "%)\n";

    std::cout << "Draws: "
        << stats.draw << " ("
        << 100.0 * stats.draw / total << "%)\n";

    if (no_draw > 0) {
        std::cout << "No-draw " << aiName(a) << " win rate: "
            << 100.0 * stats.a_win / no_draw << "%\n";

        std::cout << "No-draw " << aiName(b) << " win rate: "
            << 100.0 * stats.b_win / no_draw << "%\n";
    }

    if (stats.a_moves > 0) {
        std::cout << aiName(a) << " average search time: "
            << stats.a_time_ns / 1000000.0 / stats.a_moves
            << " ms/move\n";
    }

    if (stats.b_moves > 0) {
        std::cout << aiName(b) << " average search time: "
            << stats.b_time_ns / 1000000.0 / stats.b_moves
            << " ms/move\n";
    }
}


// ==================== 기존 ConnectFourState 구현 ====================

ConnectFourState::ConnectFourState() {}

bool ConnectFourState::isDone() const
{
    return winning_status_ != WinningStatus::NONE;
}

inline int countRun(
    const int board[H][W],
    int y,
    int x,
    int dy,
    int dx
)
{
    int count = 0;

    while (
        y >= 0 && y < H &&
        x >= 0 && x < W &&
        board[y][x] == 1
        ) {
        count++;
        y += dy;
        x += dx;
    }

    return count;
}

void ConnectFourState::advance(const int action)
{
    if (action < 0 || action >= W) {
        throw std::out_of_range("action이 0~6 범위를 벗어났습니다.");
    }

    std::pair<int, int> coordinate(-1, -1);

    for (int y = 0; y < H; ++y) {
        if (
            my_board_[y][action] == 0 &&
            enemy_board_[y][action] == 0
            ) {
            my_board_[y][action] = 1;
            coordinate = { y, action };
            break;
        }
    }

    if (coordinate.first < 0) {
        throw std::runtime_error("가득 찬 열에 돌을 놓으려고 했습니다.");
    }

    const int y0 = coordinate.first;
    const int x0 = coordinate.second;

    const auto has4 = [&](int dy, int dx) {
        const int count =
            countRun(my_board_, y0, x0, dy, dx)
            + countRun(my_board_, y0, x0, -dy, -dx)
            - 1;

        return count >= 4;
        };

    const bool win_now =
        has4(0, 1) ||
        has4(1, 0) ||
        has4(1, 1) ||
        has4(1, -1);

    bool board_full = true;

    for (int x = 0; x < W; ++x) {
        if (
            my_board_[H - 1][x] == 0 &&
            enemy_board_[H - 1][x] == 0
            ) {
            board_full = false;
            break;
        }
    }

    // 다음 플레이어 관점으로 전환
    std::swap(my_board_, enemy_board_);
    is_first_ = !is_first_;

    if (win_now) {
        winning_status_ = WinningStatus::LOSE;
    }
    else if (board_full) {
        winning_status_ = WinningStatus::DRAW;
    }
    else {
        winning_status_ = WinningStatus::NONE;
    }
}

std::vector<int> ConnectFourState::legalActions() const
{
    std::vector<int> actions;

    for (int x = 0; x < W; ++x) {
        if (
            my_board_[H - 1][x] == 0 &&
            enemy_board_[H - 1][x] == 0
            ) {
            actions.push_back(x);
        }
    }

    return actions;
}

WinningStatus ConnectFourState::getWinningStatus() const
{
    return winning_status_;
}

std::string ConnectFourState::toString() const
{
    std::stringstream ss;
    ss << "is_first:\t" << is_first_ << '\n';

    for (int y = H - 1; y >= 0; --y) {
        for (int x = 0; x < W; ++x) {
            char c = '.';

            if (my_board_[y][x] == 1) {
                c = is_first_ ? 'x' : 'o';
            }
            else if (enemy_board_[y][x] == 1) {
                c = is_first_ ? 'o' : 'x';
            }

            ss << c;
        }

        ss << '\n';
    }

    return ss.str();
}


// ==================== 실행할 대진 ====================

int main()
{
    std::string error;

    if (!mlp_policy.load("mlp_policy.bin", &error)) {
        std::cerr << error << '\n';
        return 1;
    }

    if (!cnn_policy.load("cnn_policy.bin", &error)) {
        std::cerr << error << '\n';
        return 1;
    }

    policy_mcts_config.budget_mode = SearchBudgetMode::FixedTime;
    policy_mcts_config.time_limit_ms = TIME_LIMIT_MS;
    policy_mcts_config.c_puct = 1.4;

    // 필요 없는 대진은 앞에 //를 붙이면 된다.
    //runMatch(AIType::MLPMCTS, AIType::AlphaBeta);
    runMatch(AIType::MLPMCTS, AIType::MCTS);
    //runMatch(AIType::MLPMCTS, AIType::PureMC);

    //runMatch(AIType::CNNMCTS, AIType::AlphaBeta);
    runMatch(AIType::CNNMCTS, AIType::MCTS);
    //runMatch(AIType::CNNMCTS, AIType::PureMC);

    // 둘끼리 다시 비교할 때 주석 해제
    runMatch(AIType::MLPMCTS, AIType::CNNMCTS);

    return 0;
}