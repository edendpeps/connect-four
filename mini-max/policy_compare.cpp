#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ConnectFourState.hpp"
#include "MLPPolicy.hpp"
#include "CNNPolicy.hpp"

constexpr int MATCH_PAIRS = 500;       // 실제 대국 수는 MATCH_PAIRS * 2
constexpr int OPENING_MIN_PLIES = 2;    // 랜덤 오프닝 최소 수
constexpr int OPENING_MAX_PLIES = 6;    // 랜덤 오프닝 최대 수
constexpr unsigned int RANDOM_SEED = 42u;

namespace {
    
struct Stats {
    int mlp_win = 0;
    int cnn_win = 0;
    int draw = 0;
    long long mlp_inference_ns = 0;
    long long cnn_inference_ns = 0;
    long long mlp_moves = 0;
    long long cnn_moves = 0;
    long long total_moves = 0;
};

enum class Winner {
    MLP,
    CNN,
    Draw
};

struct GameResult {
    Winner winner = Winner::Draw;
    int moves_after_opening = 0;
    long long mlp_ns = 0;
    long long cnn_ns = 0;
    int mlp_moves = 0;
    int cnn_moves = 0;
};

inline int run(const int board[H][W], int y, int x, int dy, int dx) {
    int count = 0;
    while (y >= 0 && y < H && x >= 0 && x < W && board[y][x] == 1) {
        ++count;
        y += dy;
        x += dx;
    }
    return count;
}

ConnectFourState makeRandomOpening(
    std::mt19937& rng,
    int opening_plies
) {
    // 매우 드물게 오프닝 도중 게임이 끝나면 새로 생성한다.
    for (;;) {
        ConnectFourState state;
        bool valid = true;

        for (int i = 0; i < opening_plies; ++i) {
            const auto legal = state.legalActions();
            if (legal.empty()) {
                valid = false;
                break;
            }

            std::uniform_int_distribution<int> pick(0, static_cast<int>(legal.size()) - 1);
            state.advance(legal[static_cast<std::size_t>(pick(rng))]);

            if (state.isDone()) {
                valid = false;
                break;
            }
        }

        if (valid) return state;
    }
}

const char* winnerName(Winner winner) {
    switch (winner) {
    case Winner::MLP: return "MLP";
    case Winner::CNN: return "CNN";
    default: return "DRAW";
    }
}

GameResult playGame(
    const ConnectFourState& initial_state,
    bool mlp_moves_first,
    const MLPPolicy& mlp,
    const CNNPolicy& cnn
) {
    ConnectFourState state = initial_state;
    bool mlp_turn = mlp_moves_first;
    bool last_move_by_mlp = false;
    GameResult result;

    while (!state.isDone()) {
        int action = -1;
        const auto begin = std::chrono::steady_clock::now();

        if (mlp_turn) {
            action = mlp.selectAction(state);
        }
        else {
            action = cnn.selectAction(state);
        }

        const auto end = std::chrono::steady_clock::now();
        const long long elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - begin
        ).count();

        if (mlp_turn) {
            result.mlp_ns += elapsed;
            result.mlp_moves++;
        }
        else {
            result.cnn_ns += elapsed;
            result.cnn_moves++;
        }

        const auto legal = state.legalActions();
        if (std::find(legal.begin(), legal.end(), action) == legal.end()) {
            throw std::runtime_error(
                std::string(mlp_turn ? "MLP" : "CNN")
                + "가 불가능한 수를 선택했습니다: " + std::to_string(action)
            );
        }

        state.advance(action);
        last_move_by_mlp = mlp_turn;
        mlp_turn = !mlp_turn;
        result.moves_after_opening++;
    }

    if (state.getWinningStatus() == WinningStatus::DRAW) {
        result.winner = Winner::Draw;
    }
    else if (state.getWinningStatus() == WinningStatus::LOSE) {
        // advance() 후 다음 플레이어 관점에서 LOSE이므로 방금 둔 쪽이 승리
        result.winner = last_move_by_mlp ? Winner::MLP : Winner::CNN;
    }
    else {
        // 현재 구현에서는 사실상 나오지 않지만 안전 처리
        result.winner = last_move_by_mlp ? Winner::CNN : Winner::MLP;
    }

    return result;
}

void addResult(Stats& stats, const GameResult& result) {
    if (result.winner == Winner::MLP) stats.mlp_win++;
    else if (result.winner == Winner::CNN) stats.cnn_win++;
    else stats.draw++;

    stats.mlp_inference_ns += result.mlp_ns;
    stats.cnn_inference_ns += result.cnn_ns;
    stats.mlp_moves += result.mlp_moves;
    stats.cnn_moves += result.cnn_moves;
    stats.total_moves += result.moves_after_opening;
}
} // namespace

// 기존 connectfour.cpp 대신 이 파일을 빌드할 때 필요한 게임 상태 구현부
ConnectFourState::ConnectFourState() {}

bool ConnectFourState::isDone() const {
    return winning_status_ != WinningStatus::NONE;
}

void ConnectFourState::advance(const int action) {
    if (action < 0 || action >= W) {
        throw std::out_of_range("action이 0~6 범위를 벗어났습니다.");
    }

    std::pair<int, int> coordinate(-1, -1);
    for (int y = 0; y < H; ++y) {
        if (my_board_[y][action] == 0 && enemy_board_[y][action] == 0) {
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
        const int count = run(my_board_, y0, x0, dy, dx)
            + run(my_board_, y0, x0, -dy, -dx) - 1;
        return count >= 4;
    };

    const bool win_now =
        has4(0, 1) || has4(1, 0) || has4(1, 1) || has4(1, -1);

    bool board_full = true;
    for (int x = 0; x < W; ++x) {
        if (my_board_[H - 1][x] == 0 && enemy_board_[H - 1][x] == 0) {
            board_full = false;
            break;
        }
    }

    std::swap(my_board_, enemy_board_);
    is_first_ = !is_first_;

    if (win_now) winning_status_ = WinningStatus::LOSE;
    else if (board_full) winning_status_ = WinningStatus::DRAW;
    else winning_status_ = WinningStatus::NONE;
}

std::vector<int> ConnectFourState::legalActions() const {
    std::vector<int> actions;
    for (int x = 0; x < W; ++x) {
        if (my_board_[H - 1][x] == 0 && enemy_board_[H - 1][x] == 0) {
            actions.push_back(x);
        }
    }
    return actions;
}

WinningStatus ConnectFourState::getWinningStatus() const {
    return winning_status_;
}

std::string ConnectFourState::toString() const {
    std::stringstream ss;
    ss << "is_first:\t" << is_first_ << "\n";

    for (int y = H - 1; y >= 0; --y) {
        for (int x = 0; x < W; ++x) {
            char c = '.';
            if (my_board_[y][x] == 1) c = is_first_ ? 'x' : 'o';
            else if (enemy_board_[y][x] == 1) c = is_first_ ? 'o' : 'x';
            ss << c;
        }
        ss << '\n';
    }
    return ss.str();
}

int main() {
    // 실행 파일 기준 상대 경로. 필요하면 절대 경로로 변경.
    const std::string mlp_weight_path = "mlp_policy.bin";
    const std::string cnn_weight_path = "cnn_policy.bin";

    MLPPolicy mlp;
    CNNPolicy cnn;
    std::string error;

    if (!mlp.load(mlp_weight_path, &error)) {
        std::cerr << error << '\n';
        return 1;
    }
    if (!cnn.load(cnn_weight_path, &error)) {
        std::cerr << error << '\n';
        return 1;
    }

    std::mt19937 rng(RANDOM_SEED);
    std::uniform_int_distribution<int> opening_length(
        OPENING_MIN_PLIES,
        OPENING_MAX_PLIES
    );

    std::ofstream csv("policy_battle_results.csv");
    if (!csv.is_open()) {
        std::cerr << "policy_battle_results.csv를 만들 수 없습니다.\n";
        return 1;
    }
    csv << "pair_id,game_in_pair,opening_plies,mlp_moves_first,winner,"
        << "moves_after_opening,mlp_inference_us,cnn_inference_us\n";

    Stats stats;

    for (int pair_id = 0; pair_id < MATCH_PAIRS; ++pair_id) {
        const int opening_plies = opening_length(rng);
        const ConnectFourState opening = makeRandomOpening(rng, opening_plies);

        // 같은 시작 상태에서 선후공만 교환하여 두 판 진행
        for (int game = 0; game < 2; ++game) {
            const bool mlp_moves_first = (game == 0);
            const GameResult result = playGame(opening, mlp_moves_first, mlp, cnn);
            addResult(stats, result);

            csv << pair_id << ','
                << game << ','
                << opening_plies << ','
                << (mlp_moves_first ? 1 : 0) << ','
                << winnerName(result.winner) << ','
                << result.moves_after_opening << ','
                << std::fixed << std::setprecision(3)
                << (result.mlp_ns / 1000.0) << ','
                << (result.cnn_ns / 1000.0) << '\n';
        }

        if ((pair_id + 1) % 50 == 0) {
            std::cout << "진행: " << (pair_id + 1) * 2
                << " / " << MATCH_PAIRS * 2 << " games\n";
        }
    }

    const int total_games = stats.mlp_win + stats.cnn_win + stats.draw;
    const int decisive_games = stats.mlp_win + stats.cnn_win;

    std::cout << "\n========== MLP vs CNN Policy Battle ==========\n";
    std::cout << "Total games: " << total_games << '\n';
    std::cout << "MLP wins: " << stats.mlp_win
        << " (" << std::fixed << std::setprecision(2)
        << 100.0 * stats.mlp_win / total_games << "%)\n";
    std::cout << "CNN wins: " << stats.cnn_win
        << " (" << 100.0 * stats.cnn_win / total_games << "%)\n";
    std::cout << "Draws: " << stats.draw
        << " (" << 100.0 * stats.draw / total_games << "%)\n";

    if (decisive_games > 0) {
        std::cout << "\nNo-draw MLP win rate: "
            << 100.0 * stats.mlp_win / decisive_games << "%\n";
        std::cout << "No-draw CNN win rate: "
            << 100.0 * stats.cnn_win / decisive_games << "%\n";
    }

    if (stats.mlp_moves > 0) {
        std::cout << "\nMLP average inference: "
            << stats.mlp_inference_ns / 1000.0 / stats.mlp_moves << " us/move\n";
    }
    if (stats.cnn_moves > 0) {
        std::cout << "CNN average inference: "
            << stats.cnn_inference_ns / 1000.0 / stats.cnn_moves << " us/move\n";
    }

    std::cout << "Average moves after opening: "
        << static_cast<double>(stats.total_moves) / total_games << '\n';
    std::cout << "Detailed result: policy_battle_results.csv\n";

    return 0;
}
