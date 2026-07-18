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
#include "PolicyMCTS.hpp"

// =========================
// Experiment settings
// =========================
constexpr int MATCH_PAIRS = 100;        // 100 pairs = 200 games
constexpr int OPENING_MIN_PLIES = 2;
constexpr int OPENING_MAX_PLIES = 6;
constexpr unsigned int RANDOM_SEED = 42u;

// true  : same number of MCTS simulations per move (policy quality comparison)
// false : same time limit per move (practical speed comparison)
constexpr bool USE_FIXED_SIMULATIONS = false;
constexpr int SIMULATIONS_PER_MOVE = 40;
constexpr int TIME_LIMIT_MS = 750;
constexpr double C_PUCT = 1.4;

namespace {
using Clock = std::chrono::steady_clock;

struct SideStats {
    int wins = 0;
    long long moves = 0;
    long long search_ns = 0;
    long long simulations = 0;
    long long policy_evaluations = 0;
    long long policy_inference_ns = 0;
};

struct Stats {
    SideStats mlp;
    SideStats cnn;
    int draws = 0;
    long long total_moves = 0;
};

enum class Winner {
    MLP_MCTS,
    CNN_MCTS,
    Draw
};

struct GameResult {
    Winner winner = Winner::Draw;
    int moves_after_opening = 0;

    long long mlp_search_ns = 0;
    long long cnn_search_ns = 0;
    long long mlp_moves = 0;
    long long cnn_moves = 0;
    long long mlp_simulations = 0;
    long long cnn_simulations = 0;
    long long mlp_policy_evaluations = 0;
    long long cnn_policy_evaluations = 0;
    long long mlp_policy_ns = 0;
    long long cnn_policy_ns = 0;
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

ConnectFourState makeRandomOpening(std::mt19937& rng, int opening_plies) {
    for (;;) {
        ConnectFourState state;
        bool valid = true;

        for (int i = 0; i < opening_plies; ++i) {
            const auto legal = state.legalActions();
            if (legal.empty()) {
                valid = false;
                break;
            }

            std::uniform_int_distribution<int> pick(
                0,
                static_cast<int>(legal.size()) - 1
            );
            state.advance(legal[static_cast<std::size_t>(pick(rng))]);

            if (state.isDone()) {
                valid = false;
                break;
            }
        }

        if (valid) return state;
    }
}

bool isLegalAction(const ConnectFourState& state, int action) {
    const auto legal = state.legalActions();
    return std::find(legal.begin(), legal.end(), action) != legal.end();
}

int fallbackAction(const ConnectFourState& state) {
    const auto legal = state.legalActions();
    if (legal.empty()) return -1;

    constexpr int order[7] = { 3, 2, 4, 1, 5, 0, 6 };
    for (int action : order) {
        if (std::find(legal.begin(), legal.end(), action) != legal.end()) {
            return action;
        }
    }
    return legal.front();
}

const char* winnerName(Winner winner) {
    switch (winner) {
    case Winner::MLP_MCTS: return "MLP_MCTS";
    case Winner::CNN_MCTS: return "CNN_MCTS";
    default: return "DRAW";
    }
}

GameResult playGame(
    const ConnectFourState& initial_state,
    bool mlp_moves_first,
    const MLPPolicy& mlp,
    const CNNPolicy& cnn,
    const PolicyMCTSConfig& config,
    unsigned int game_seed
) {
    ConnectFourState state = initial_state;
    bool mlp_turn = mlp_moves_first;
    bool last_move_by_mlp = false;
    GameResult result;
    std::mt19937 rng(game_seed);

    while (!state.isDone()) {
        const auto begin = Clock::now();
        PolicyMCTSResult search_result;

        if (mlp_turn) {
            search_result = MLPMCTSAction(state, mlp, config, rng);
        }
        else {
            search_result = CNNMCTSAction(state, cnn, config, rng);
        }

        const auto end = Clock::now();
        const long long elapsed_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

        int action = search_result.action;
        if (!isLegalAction(state, action)) {
            action = fallbackAction(state);
        }
        if (action < 0) break;

        if (mlp_turn) {
            result.mlp_search_ns += elapsed_ns;
            result.mlp_moves++;
            result.mlp_simulations += search_result.simulations;
            result.mlp_policy_evaluations += search_result.policy_evaluations;
            result.mlp_policy_ns += search_result.policy_inference_ns;
        }
        else {
            result.cnn_search_ns += elapsed_ns;
            result.cnn_moves++;
            result.cnn_simulations += search_result.simulations;
            result.cnn_policy_evaluations += search_result.policy_evaluations;
            result.cnn_policy_ns += search_result.policy_inference_ns;
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
        result.winner = last_move_by_mlp ? Winner::MLP_MCTS : Winner::CNN_MCTS;
    }
    else {
        // Safety fallback for an unexpected WIN representation.
        result.winner = last_move_by_mlp ? Winner::CNN_MCTS : Winner::MLP_MCTS;
    }

    return result;
}

void addResult(Stats& stats, const GameResult& result) {
    if (result.winner == Winner::MLP_MCTS) stats.mlp.wins++;
    else if (result.winner == Winner::CNN_MCTS) stats.cnn.wins++;
    else stats.draws++;

    stats.mlp.moves += result.mlp_moves;
    stats.mlp.search_ns += result.mlp_search_ns;
    stats.mlp.simulations += result.mlp_simulations;
    stats.mlp.policy_evaluations += result.mlp_policy_evaluations;
    stats.mlp.policy_inference_ns += result.mlp_policy_ns;

    stats.cnn.moves += result.cnn_moves;
    stats.cnn.search_ns += result.cnn_search_ns;
    stats.cnn.simulations += result.cnn_simulations;
    stats.cnn.policy_evaluations += result.cnn_policy_evaluations;
    stats.cnn.policy_inference_ns += result.cnn_policy_ns;

    stats.total_moves += result.moves_after_opening;
}

void printSideStats(const char* name, const SideStats& stats) {
    if (stats.moves <= 0) return;

    std::cout << name << " average search time: "
        << stats.search_ns / 1'000'000.0 / stats.moves << " ms/move\n";
    std::cout << name << " average simulations: "
        << static_cast<double>(stats.simulations) / stats.moves << " /move\n";
    std::cout << name << " average policy evaluations: "
        << static_cast<double>(stats.policy_evaluations) / stats.moves << " /move\n";

    if (stats.policy_evaluations > 0) {
        std::cout << name << " average policy inference: "
            << stats.policy_inference_ns / 1000.0 / stats.policy_evaluations
            << " us/evaluation\n";
    }
}
} // namespace

// =========================
// ConnectFourState implementation
// Build this file instead of the old connectfour.cpp/policy_compare.cpp.
// =========================
ConnectFourState::ConnectFourState() {}

bool ConnectFourState::isDone() const {
    return winning_status_ != WinningStatus::NONE;
}

void ConnectFourState::advance(const int action) {
    if (action < 0 || action >= W) {
        throw std::out_of_range("action is outside 0..6");
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
        throw std::runtime_error("attempted to play in a full column");
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

    PolicyMCTSConfig config;
    config.budget_mode = USE_FIXED_SIMULATIONS
        ? SearchBudgetMode::FixedSimulations
        : SearchBudgetMode::FixedTime;
    config.simulations = SIMULATIONS_PER_MOVE;
    config.time_limit_ms = TIME_LIMIT_MS;
    config.c_puct = C_PUCT;

    std::cout << "========== MLP+MCTS vs CNN+MCTS ==========\n";
    std::cout << "Match pairs: " << MATCH_PAIRS
        << " (total games: " << MATCH_PAIRS * 2 << ")\n";
    if (USE_FIXED_SIMULATIONS) {
        std::cout << "Budget: " << SIMULATIONS_PER_MOVE
            << " simulations per move\n";
    }
    else {
        std::cout << "Budget: " << TIME_LIMIT_MS << " ms per move\n";
    }
    std::cout << "C_PUCT: " << C_PUCT << "\n\n";

    std::mt19937 opening_rng(RANDOM_SEED);
    std::uniform_int_distribution<int> opening_length(
        OPENING_MIN_PLIES,
        OPENING_MAX_PLIES
    );

    std::ofstream csv("policy_mcts_battle_results.csv");
    if (!csv.is_open()) {
        std::cerr << "cannot create policy_mcts_battle_results.csv\n";
        return 1;
    }

    csv << "pair_id,game_in_pair,opening_plies,mlp_moves_first,winner,"
        << "moves_after_opening,mlp_search_ms,cnn_search_ms,"
        << "mlp_simulations,cnn_simulations,"
        << "mlp_policy_evals,cnn_policy_evals\n";

    Stats stats;

    for (int pair_id = 0; pair_id < MATCH_PAIRS; ++pair_id) {
        const int opening_plies = opening_length(opening_rng);
        const ConnectFourState opening = makeRandomOpening(opening_rng, opening_plies);

        for (int game = 0; game < 2; ++game) {
            const bool mlp_moves_first = (game == 0);
            const unsigned int game_seed = RANDOM_SEED
                ^ static_cast<unsigned int>(pair_id * 2 + game + 1) * 0x9E3779B9u;

            const GameResult result = playGame(
                opening,
                mlp_moves_first,
                mlp,
                cnn,
                config,
                game_seed
            );
            addResult(stats, result);

            csv << pair_id << ','
                << game << ','
                << opening_plies << ','
                << (mlp_moves_first ? 1 : 0) << ','
                << winnerName(result.winner) << ','
                << result.moves_after_opening << ','
                << std::fixed << std::setprecision(3)
                << result.mlp_search_ns / 1'000'000.0 << ','
                << result.cnn_search_ns / 1'000'000.0 << ','
                << result.mlp_simulations << ','
                << result.cnn_simulations << ','
                << result.mlp_policy_evaluations << ','
                << result.cnn_policy_evaluations << '\n';
        }

        if ((pair_id + 1) % 10 == 0) {
            std::cout << "Progress: " << (pair_id + 1) * 2
                << " / " << MATCH_PAIRS * 2 << " games\n";
        }
    }

    const int total_games = stats.mlp.wins + stats.cnn.wins + stats.draws;
    const int decisive_games = stats.mlp.wins + stats.cnn.wins;

    std::cout << "\n========== Final Result ==========\n";
    std::cout << "Total games: " << total_games << '\n';
    std::cout << "MLP+MCTS wins: " << stats.mlp.wins
        << " (" << std::fixed << std::setprecision(2)
        << 100.0 * stats.mlp.wins / total_games << "%)\n";
    std::cout << "CNN+MCTS wins: " << stats.cnn.wins
        << " (" << 100.0 * stats.cnn.wins / total_games << "%)\n";
    std::cout << "Draws: " << stats.draws
        << " (" << 100.0 * stats.draws / total_games << "%)\n";

    if (decisive_games > 0) {
        std::cout << "\nNo-draw MLP+MCTS win rate: "
            << 100.0 * stats.mlp.wins / decisive_games << "%\n";
        std::cout << "No-draw CNN+MCTS win rate: "
            << 100.0 * stats.cnn.wins / decisive_games << "%\n";
    }

    std::cout << '\n';
    printSideStats("MLP+MCTS", stats.mlp);
    std::cout << '\n';
    printSideStats("CNN+MCTS", stats.cnn);

    std::cout << "\nAverage moves after opening: "
        << static_cast<double>(stats.total_moves) / total_games << '\n';
    std::cout << "Detailed result: policy_mcts_battle_results.csv\n";

    return 0;
}
