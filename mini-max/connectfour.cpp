#include <string>
#include <array>
#include <vector>
#include <sstream>
#include <utility>
#include <random>
#include <assert.h>
#include <math.h>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <functional>
#include <queue>
#include <set>
#include <limits>
#include <fstream>

#include "ConnectFourState.hpp"
#include "minimax.hpp"
#include "Monte_Carlo.hpp"
#include "MCTS.h"

int monte_win = 0;
int minimax_win = 0;
int minimax2_win = 0;
int game_draw = 0;
int puremc_win = 0;

int game_limit = 2000;

// 상태 생성용 AI 시간 제한
// 빠르게 데이터 많이 만들고 싶으면 100~300 추천
const int game_ai_time_limit = 100;

// 정책망 라벨 teacher용 MCTS 시간 제한
// 라벨 품질 우선이면 500~1000, 시간 아끼려면 100~300
const int policy_teacher_time_limit = 500;

const int INF = 100000000;
const int minimax_depth = INF;
const int roll_out = INF;

ConnectFourState::ConnectFourState() {}

bool ConnectFourState::isDone() const {
    return winning_status_ != WinningStatus::NONE;
}

enum class OpponentType {
    AlphaBeta,
    MCTS,
    PMC
};

enum class DataSplit {
    Train,
    Val,
    Test
};

// helper: (y,x)에서 (dy,dx) 방향으로 내 돌 연속 길이
inline int run(const int board[H][W], int y, int x, int dy, int dx) {
    int cnt = 0;
    while (y >= 0 && y < H && x >= 0 && x < W && board[y][x] == 1) {
        ++cnt;
        y += dy;
        x += dx;
    }
    return cnt;
}

void ConnectFourState::advance(const int action)
{
    // 1. 돌 놓기
    std::pair<int, int> coordinate(-1, -1);
    for (int y = 0; y < H; ++y) {
        if (my_board_[y][action] == 0 && enemy_board_[y][action] == 0) {
            my_board_[y][action] = 1;
            coordinate = { y, action };
            break;
        }
    }

    int y0 = coordinate.first;
    int x0 = coordinate.second;

    auto has4 = [&](int dy, int dx) {
        int c = run(my_board_, y0, x0, dy, dx)
            + run(my_board_, y0, x0, -dy, -dx) - 1;
        return c >= 4;
        };

    bool win_now =
        has4(0, 1) || has4(1, 0) || has4(1, 1) || has4(1, -1);

    bool board_full = false;
    {
        auto acts = legalActions();
        board_full = acts.empty();
    }

    // 2. 턴 넘기기
    std::swap(my_board_, enemy_board_);
    is_first_ = !is_first_;

    // 3. 다음 플레이어 관점에서 상태 기록
    if (win_now) {
        // 방금 둔 이전 플레이어가 이겼으므로 현재 플레이어는 패배 상태
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
    for (int x = 0; x < W; x++) {
        for (int y = H - 1; y >= 0; y--) {
            if (my_board_[y][x] == 0 && enemy_board_[y][x] == 0) {
                actions.emplace_back(x);
                break;
            }
        }
    }
    return actions;
}

WinningStatus ConnectFourState::getWinningStatus() const {
    return this->winning_status_;
}

std::string ConnectFourState::toString() const
{
    std::stringstream ss("");
    ss << "is_first:\t" << this->is_first_ << "\n";
    for (int y = H - 1; y >= 0; y--)
    {
        for (int x = 0; x < W; x++)
        {
            char c = '.';
            if (my_board_[y][x] == 1)
            {
                c = (is_first_ ? 'x' : 'o');
            }
            else if (enemy_board_[y][x] == 1)
            {
                c = (is_first_ ? 'o' : 'x');
            }
            ss << c;
        }
        ss << "\n";
    }
    return ss.str();
}

using State = ConnectFourState;

// =========================
// 정책망 데이터 저장부
// CSV: c0,c1,...,c41,best_action
// c0~c41: 현재 플레이어 기준 보드
//  1 = 현재 차례 플레이어 돌
// -1 = 상대 돌
//  0 = 빈칸
// best_action: MCTS teacher가 고른 열 번호, 0~6
// =========================

static std::ofstream policy_train_file;
static std::ofstream policy_val_file;
static std::ofstream policy_test_file;
static std::ofstream* current_policy_file = nullptr;

void write_policy_header(std::ofstream& fout) {
    for (int i = 0; i < 42; i++) {
        fout << "c" << i << ",";
    }
    fout << "best_action\n";
}

void open_policy_files(
    const std::string& train_filename,
    const std::string& val_filename,
    const std::string& test_filename
) {
    policy_train_file.open(train_filename);
    policy_val_file.open(val_filename);
    policy_test_file.open(test_filename);

    if (!policy_train_file.is_open()) {
        std::cout << "policy train file open failed: " << train_filename << "\n";
    }
    if (!policy_val_file.is_open()) {
        std::cout << "policy val file open failed: " << val_filename << "\n";
    }
    if (!policy_test_file.is_open()) {
        std::cout << "policy test file open failed: " << test_filename << "\n";
    }

    write_policy_header(policy_train_file);
    write_policy_header(policy_val_file);
    write_policy_header(policy_test_file);

    current_policy_file = &policy_train_file;
}

void set_policy_split(DataSplit split) {
    if (split == DataSplit::Train) {
        current_policy_file = &policy_train_file;
    }
    else if (split == DataSplit::Val) {
        current_policy_file = &policy_val_file;
    }
    else {
        current_policy_file = &policy_test_file;
    }
}

void close_policy_files() {
    if (policy_train_file.is_open()) policy_train_file.close();
    if (policy_val_file.is_open()) policy_val_file.close();
    if (policy_test_file.is_open()) policy_test_file.close();
}

bool is_legal_action(const State& s, int action) {
    auto acts = s.legalActions();
    return std::find(acts.begin(), acts.end(), action) != acts.end();
}

int fallback_action(const State& s) {
    auto acts = s.legalActions();
    if (acts.empty()) return -1;

    // 중앙열 우선 fallback
    const int order[7] = { 3, 2, 4, 1, 5, 0, 6 };
    for (int a : order) {
        if (std::find(acts.begin(), acts.end(), a) != acts.end()) return a;
    }
    return acts.front();
}

void save_policy_row(const State& s, int best_action) {
    if (current_policy_file == nullptr || !current_policy_file->is_open()) return;
    if (!is_legal_action(s, best_action)) return;

    const int(*my)[W] = s.getMyBoard();
    const int(*opp)[W] = s.getEnemyBoard();

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int v = 0;

            if (my[y][x] == 1) v = 1;
            else if (opp[y][x] == 1) v = -1;

            (*current_policy_file) << v << ",";
        }
    }

    (*current_policy_file) << best_action << "\n";
}

bool last_move_by_minimax = false;

void playGame(bool first_is_minimax, OpponentType opponent)
{
    int turn_count = 0;
    auto state = State();

    while (!state.isDone())
    {
        bool minimax_turn = (turn_count % 2 == 0) == first_is_minimax;

        int teacher_action = -1;

        // 정책망 라벨 저장
        // 초반 빈 보드/너무 단순한 상태를 빼고 싶어서 turn_count >= 4부터 저장
        if (turn_count >= 4) {
            teacher_action = MCTSAction(state, roll_out, policy_teacher_time_limit);
            save_policy_row(state, teacher_action);
        }

        int action = -1;

        if (minimax_turn)
        {
            action = alphaBetaAction(state, minimax_depth, game_ai_time_limit);
        }
        else
        {
            if (opponent == OpponentType::AlphaBeta)
            {
                action = alphaBetaAction(state, minimax_depth, game_ai_time_limit);
            }
            else if (opponent == OpponentType::MCTS)
            {
                // 이미 teacher MCTS를 돌렸으면 그 결과 재사용해서 시간 절약
                if (is_legal_action(state, teacher_action)) {
                    action = teacher_action;
                }
                else {
                    action = MCTSAction(state, roll_out, game_ai_time_limit);
                }
            }
            else
            {
                action = MontecarloAction(state, roll_out, game_ai_time_limit);
            }
        }

        if (!is_legal_action(state, action)) {
            action = fallback_action(state);
        }

        if (action == -1) break;

        state.advance(action);
        last_move_by_minimax = minimax_turn;
        turn_count++;
    }

    if (state.getWinningStatus() == WinningStatus::DRAW)
    {
        game_draw++;
    }
    else if (state.getWinningStatus() == WinningStatus::LOSE)
    {
        // 방금 둔 사람이 승리
        if (last_move_by_minimax) {
            minimax_win++;
        }
        else
        {
            if (opponent == OpponentType::MCTS) {
                monte_win++;
            }
            else if (opponent == OpponentType::PMC) {
                puremc_win++;
            }
            else {
                minimax2_win++;
            }
        }
    }
    else if (state.getWinningStatus() == WinningStatus::WIN)
    {
        // 현재 구조에서는 거의 안 나오지만 안전용
        if (last_move_by_minimax)
        {
            if (opponent == OpponentType::MCTS) {
                monte_win++;
            }
            else if (opponent == OpponentType::PMC) {
                puremc_win++;
            }
            else {
                minimax2_win++;
            }
        }
        else {
            minimax_win++;
        }
    }
}

std::string split_name(DataSplit split) {
    if (split == DataSplit::Train) return "training";
    if (split == DataSplit::Val) return "validation";
    return "test";
}

int main()
{
    open_policy_files(
        "C:/Users/User/Desktop/policy_train.csv",
        "C:/Users/User/Desktop/policy_val.csv",
        "C:/Users/User/Desktop/policy_test.csv"
    );

    const int training_game_limit = static_cast<int>(game_limit * 0.70);
    const int validation_game_limit = static_cast<int>(game_limit * 0.15);
    const int test_game_limit = game_limit - training_game_limit - validation_game_limit;

    for (int i = 0; i < game_limit; i++) {
        DataSplit split;
        int split_index;

        if (i < training_game_limit) {
            split = DataSplit::Train;
            split_index = i;
        }
        else if (i < training_game_limit + validation_game_limit) {
            split = DataSplit::Val;
            split_index = i - training_game_limit;
        }
        else {
            split = DataSplit::Test;
            split_index = i - training_game_limit - validation_game_limit;
        }

        set_policy_split(split);

        std::cout << "\n---------------------- game_count: " << i
            << " [" << split_name(split) << "]\n";

        // 상태 생성용 상대 비율
        // 65% AlphaBeta, 25% MCTS, 10% PureMC
        OpponentType opponent;
        int r = split_index % 100;

        if (r < 65) opponent = OpponentType::AlphaBeta;
        else if (r < 90) opponent = OpponentType::MCTS;
        else opponent = OpponentType::PMC;

        bool first_is_minimax = (i % 2 == 0);
        playGame(first_is_minimax, opponent);
    }

    close_policy_files();

    std::cout << "\n========== Result ==========\n";
    std::cout << "alphabeta_win: " << minimax_win << "\n";
    std::cout << "MCTS_win: " << monte_win << "\n";
    std::cout << "alphabeta2_win: " << minimax2_win << "\n";
    std::cout << "pureMC_win: " << puremc_win << "\n";
    std::cout << "Draw: " << game_draw << "\n";

    std::cout << "\nCSV saved:\n";
    std::cout << "C:/Users/User/Desktop/policy_train.csv\n";
    std::cout << "C:/Users/User/Desktop/policy_val.csv\n";
    std::cout << "C:/Users/User/Desktop/policy_test.csv\n";

    return 0;
}