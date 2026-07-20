#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
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
#include "minimax.hpp"
#include "Monte_Carlo.hpp"
#include "MCTS.h"

#include "MLPPolicy.hpp"
#include "CNNPolicy.hpp"
#include "PolicyMCTS.hpp"
#include "FairMCTS.hpp"

// ==================== 여기만 수정 ====================
const int GAME_LIMIT = 200;          // 대진 하나당 총 판수
const int TIME_LIMIT_MS = 100;       // 한 수당 제한시간
const int INF = 100000000;
const int RANDOM_SEED = 42;

// ==================== 공정 비교 실험 설정 ====================
const bool RUN_FIXED_SIMULATIONS = true;
const bool RUN_FIXED_TIME = true;
const int FIXED_SIMULATIONS = 10000;
const int FIXED_TIME_MS = 100;
const int OPENINGS_PER_SEED = 50;
const int MIN_OPENING_PLIES = 2;
const int MAX_OPENING_PLIES = 6;
const std::array<std::uint32_t, 3> EXPERIMENT_SEEDS = {
    42u, 123u, 777u
};
const char* const GAMES_CSV = "fair_mcts_games.csv";
const char* const SUMMARY_CSV = "fair_mcts_summary.csv";
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


// ==================== 공정 MCTS 비교 실험 ====================

enum class ExperimentAI {
    PlainUCT,
    MLPRoot,
    CNNRoot
};

struct SearchTotals {
    long long moves = 0;
    long long simulations = 0;
    long long policy_evaluations = 0;
    long long policy_inference_ns = 0;
    long long search_time_ns = 0;

    void add(const FairMCTSResult& result, long long elapsed_ns)
    {
        ++moves;
        simulations += result.simulations;
        policy_evaluations += result.policy_evaluations;
        policy_inference_ns += result.policy_inference_ns;
        search_time_ns += elapsed_ns;
    }
};

struct ExperimentGameResult {
    // 0: A 승, 1: B 승, 2: 무승부
    int winner = 2;
    int moves = 0;
    SearchTotals a;
    SearchTotals b;
};

struct ExperimentSummary {
    int games = 0;
    int a_wins = 0;
    int b_wins = 0;
    int draws = 0;
    SearchTotals a;
    SearchTotals b;
};

struct OpeningCase {
    State state;
    int plies = 0;
};

struct Matchup {
    ExperimentAI a;
    ExperimentAI b;
};

const char* experimentAIName(ExperimentAI ai)
{
    switch (ai) {
    case ExperimentAI::PlainUCT: return "UCT_MCTS";
    case ExperimentAI::MLPRoot:  return "MLP_ROOT_MCTS";
    case ExperimentAI::CNNRoot:  return "CNN_ROOT_MCTS";
    default:                      return "UNKNOWN";
    }
}

std::uint32_t deriveSeed(std::uint32_t base, std::uint32_t salt)
{
    std::uint32_t value = base ^ (salt + 0x9e3779b9u
        + (base << 6) + (base >> 2));
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

OpeningCase makeRandomOpening(std::uint32_t seed)
{
    OpeningCase opening;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> plies_dist(
        MIN_OPENING_PLIES,
        MAX_OPENING_PLIES
    );
    opening.plies = plies_dist(rng);

    for (int ply = 0; ply < opening.plies; ++ply) {
        const auto legal = opening.state.legalActions();
        if (legal.empty() || opening.state.isDone()) {
            throw std::runtime_error("랜덤 오프닝 생성 중 게임이 종료되었습니다.");
        }

        std::uniform_int_distribution<int> action_dist(
            0,
            static_cast<int>(legal.size()) - 1
        );
        opening.state.advance(
            legal[static_cast<std::size_t>(action_dist(rng))]
        );
    }

    return opening;
}

FairMCTSResult runExperimentSearch(
    ExperimentAI ai,
    const State& state,
    const FairMCTSConfig& config,
    std::mt19937& rng
)
{
    switch (ai) {
    case ExperimentAI::PlainUCT:
        return PlainMCTSAction(state, config, rng);
    case ExperimentAI::MLPRoot:
        return MLPRootMCTSAction(state, mlp_policy, config, rng);
    case ExperimentAI::CNNRoot:
        return CNNRootMCTSAction(state, cnn_policy, config, rng);
    }

    return FairMCTSResult{};
}

ExperimentGameResult playExperimentGame(
    const State& opening,
    ExperimentAI a,
    ExperimentAI b,
    bool a_controls_current_player,
    const FairMCTSConfig& config,
    std::uint32_t game_seed
)
{
    State state = opening;
    ExperimentGameResult game;
    bool a_turn = a_controls_current_player;
    bool last_move_by_a = false;

    while (!state.isDone()) {
        const ExperimentAI current_ai = a_turn ? a : b;

        // 매 수마다 RNG를 새로 만들어 이전 탐색의 난수 소비와 분리한다.
        const std::uint32_t move_seed = deriveSeed(
            game_seed,
            static_cast<std::uint32_t>(game.moves + 1)
        );
        std::mt19937 move_rng(move_seed);

        const auto begin = std::chrono::steady_clock::now();
        const FairMCTSResult search = runExperimentSearch(
            current_ai,
            state,
            config,
            move_rng
        );
        const auto end = std::chrono::steady_clock::now();
        const long long elapsed_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - begin
            ).count();

        const auto legal = state.legalActions();
        if (std::find(legal.begin(), legal.end(), search.action)
            == legal.end()) {
            throw std::runtime_error(
                std::string(experimentAIName(current_ai))
                + "가 불가능한 수를 선택했습니다: "
                + std::to_string(search.action)
            );
        }

        if (a_turn) game.a.add(search, elapsed_ns);
        else game.b.add(search, elapsed_ns);

        state.advance(search.action);
        last_move_by_a = a_turn;
        a_turn = !a_turn;
        ++game.moves;
    }

    if (state.getWinningStatus() == WinningStatus::DRAW) {
        game.winner = 2;
    }
    else if (state.getWinningStatus() == WinningStatus::LOSE) {
        // advance()가 관점을 교환하므로 LOSE는 방금 둔 쪽의 승리다.
        game.winner = last_move_by_a ? 0 : 1;
    }
    else {
        game.winner = last_move_by_a ? 1 : 0;
    }

    return game;
}

void addGameToSummary(
    ExperimentSummary& summary,
    const ExperimentGameResult& game
)
{
    ++summary.games;
    if (game.winner == 0) ++summary.a_wins;
    else if (game.winner == 1) ++summary.b_wins;
    else ++summary.draws;

    summary.a.moves += game.a.moves;
    summary.a.simulations += game.a.simulations;
    summary.a.policy_evaluations += game.a.policy_evaluations;
    summary.a.policy_inference_ns += game.a.policy_inference_ns;
    summary.a.search_time_ns += game.a.search_time_ns;

    summary.b.moves += game.b.moves;
    summary.b.simulations += game.b.simulations;
    summary.b.policy_evaluations += game.b.policy_evaluations;
    summary.b.policy_inference_ns += game.b.policy_inference_ns;
    summary.b.search_time_ns += game.b.search_time_ns;
}

double average(long long total, long long count)
{
    return count == 0
        ? 0.0
        : static_cast<double>(total) / static_cast<double>(count);
}

void writeGameRow(
    std::ofstream& csv,
    const char* budget_name,
    const Matchup& matchup,
    std::uint32_t base_seed,
    int opening_id,
    int leg,
    const OpeningCase& opening,
    const ExperimentGameResult& game
)
{
    const bool a_first = leg == 0;
    const char* winner = game.winner == 0
        ? experimentAIName(matchup.a)
        : (game.winner == 1 ? experimentAIName(matchup.b) : "DRAW");

    csv << budget_name << ','
        << experimentAIName(matchup.a) << "_vs_"
        << experimentAIName(matchup.b) << ','
        << base_seed << ',' << opening_id << ',' << leg << ','
        << opening.plies << ','
        << experimentAIName(a_first ? matchup.a : matchup.b) << ','
        << experimentAIName(a_first ? matchup.b : matchup.a) << ','
        << winner << ',' << game.moves << ','
        << game.a.moves << ',' << game.b.moves << ','
        << game.a.simulations << ',' << game.b.simulations << ','
        << game.a.policy_evaluations << ','
        << game.b.policy_evaluations << ','
        << game.a.policy_inference_ns << ','
        << game.b.policy_inference_ns << ','
        << game.a.search_time_ns << ','
        << game.b.search_time_ns << '\n';
}

void writeSummaryRow(
    std::ofstream& csv,
    const char* budget_name,
    const Matchup& matchup,
    std::uint32_t base_seed,
    const ExperimentSummary& summary
)
{
    csv << budget_name << ','
        << experimentAIName(matchup.a) << "_vs_"
        << experimentAIName(matchup.b) << ','
        << base_seed << ',' << summary.games << ','
        << summary.a_wins << ',' << summary.b_wins << ','
        << summary.draws << ','
        << average(summary.a.simulations, summary.a.moves) << ','
        << average(summary.b.simulations, summary.b.moves) << ','
        << average(summary.a.policy_inference_ns,
            summary.a.policy_evaluations) << ','
        << average(summary.b.policy_inference_ns,
            summary.b.policy_evaluations) << ','
        << average(summary.a.search_time_ns, summary.a.moves) << ','
        << average(summary.b.search_time_ns, summary.b.moves) << '\n';
}

void runExperimentBudget(
    const char* budget_name,
    FairSearchBudgetMode budget_mode,
    std::ofstream& games_csv,
    std::ofstream& summary_csv
)
{
    FairMCTSConfig config;
    config.budget_mode = budget_mode;
    config.simulations = FIXED_SIMULATIONS;
    config.time_limit_ms = FIXED_TIME_MS;

    const std::array<Matchup, 3> matchups = {{
        { ExperimentAI::PlainUCT, ExperimentAI::MLPRoot },
        { ExperimentAI::PlainUCT, ExperimentAI::CNNRoot },
        { ExperimentAI::MLPRoot, ExperimentAI::CNNRoot }
    }};

    for (const Matchup& matchup : matchups) {
        for (std::uint32_t base_seed : EXPERIMENT_SEEDS) {
            ExperimentSummary summary;

            for (int opening_id = 0;
                opening_id < OPENINGS_PER_SEED;
                ++opening_id) {
                const std::uint32_t opening_seed = deriveSeed(
                    base_seed,
                    static_cast<std::uint32_t>(opening_id + 1)
                );
                const OpeningCase opening = makeRandomOpening(opening_seed);

                // 같은 opening.state를 복사하고 현재 플레이어 담당만 교환한다.
                for (int leg = 0; leg < 2; ++leg) {
                    const std::uint32_t game_seed = deriveSeed(
                        opening_seed,
                        static_cast<std::uint32_t>(1000 + leg)
                    );
                    const ExperimentGameResult game = playExperimentGame(
                        opening.state,
                        matchup.a,
                        matchup.b,
                        leg == 0,
                        config,
                        game_seed
                    );

                    writeGameRow(
                        games_csv,
                        budget_name,
                        matchup,
                        base_seed,
                        opening_id,
                        leg,
                        opening,
                        game
                    );
                    addGameToSummary(summary, game);
                }
            }

            writeSummaryRow(
                summary_csv,
                budget_name,
                matchup,
                base_seed,
                summary
            );

            std::cout << budget_name << ": "
                << experimentAIName(matchup.a) << " vs "
                << experimentAIName(matchup.b) << ", seed="
                << base_seed << " 완료\n";
        }
    }
}

int runFairExperiments()
{
    std::cout << "공정 MCTS 비교 실험을 시작합니다.\n" << std::flush;

    std::ofstream games_csv(GAMES_CSV);
    std::ofstream summary_csv(SUMMARY_CSV);
    if (!games_csv.is_open() || !summary_csv.is_open()) {
        std::cerr << "실험 CSV 파일을 만들 수 없습니다.\n";
        return 1;
    }

    games_csv
        << "budget,matchup,base_seed,opening_id,leg,opening_plies,"
        << "first_ai,second_ai,winner,moves,a_moves,b_moves,"
        << "a_simulations,b_simulations,a_policy_evaluations,"
        << "b_policy_evaluations,a_policy_inference_ns,"
        << "b_policy_inference_ns,a_search_time_ns,b_search_time_ns\n";

    summary_csv
        << "budget,matchup,base_seed,games,a_wins,b_wins,draws,"
        << "a_avg_simulations,b_avg_simulations,"
        << "a_avg_policy_inference_ns,b_avg_policy_inference_ns,"
        << "a_avg_search_time_ns,b_avg_search_time_ns\n";

    if (RUN_FIXED_SIMULATIONS) {
        runExperimentBudget(
            "fixed_simulations",
            FairSearchBudgetMode::FixedSimulations,
            games_csv,
            summary_csv
        );
    }

    if (RUN_FIXED_TIME) {
        runExperimentBudget(
            "fixed_time_100ms",
            FairSearchBudgetMode::FixedTime,
            games_csv,
            summary_csv
        );
    }

    std::cout << "상세 결과: " << GAMES_CSV << '\n';
    std::cout << "요약 결과: " << SUMMARY_CSV << '\n';
    return 0;
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

    const int experiment_result = runFairExperiments();

#if 0
    // 기존 단순 대진 실행부. 필요하면 실험 호출과 바꿔서 사용할 수 있다.
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

#endif

    return experiment_result;
}
