#include <algorithm>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <chrono>
#include <fstream>
#include "ConnectFourState.hpp"
#include "minimax.hpp"

using State = ConnectFourState;

static constexpr int INF = 100000000;
static constexpr int WIN_SCORE = 100000000;
static constexpr int THREE_OPEN = 1000;
static constexpr int THREE_OPEN_BLOCK = 2000;
static constexpr int TWO_OPEN = 50;
static constexpr int CENTER_BONUS_PIECE = 6;
double duration;

// ── 킬러 휴리스틱 테이블 ──────────────────────────────────────
// 각 탐색 층(ply)마다 베타 컷오프를 낸 수를 슬롯 2개에 기억
// 커넥트포 최대 수순 = 42수, 여유분 포함 50으로 설정
static constexpr int MAX_PLY = 50;
static constexpr int MAX_KILLERS = 2;

using KillerTable = std::array<std::array<int, MAX_KILLERS>, MAX_PLY>;

// 킬러 등록: 슬롯 0이 가장 최신, 슬롯 1은 이전 킬러
// 이미 등록된 수는 중복 저장하지 않음
static inline void store_killer(KillerTable& kt, int ply, int action) {
    auto& slot = kt[ply];
    if (slot[0] == action || slot[1] == action) return;
    slot[1] = slot[0]; // 기존 킬러를 한 칸 밀어냄
    slot[0] = action;  // 새 킬러를 슬롯 0에
}

// ── 보드 직접 접근 헬퍼 ──────────────────────────────────────
static inline int score_window_direct(
    const int my[H][W], const int opp[H][W],
    int y0, int x0, int dy, int dx)
{
    int my_cnt = 0, opp_cnt = 0, emp = 0;
    for (int k = 0; k < 4; k++) {
        int y = y0 + k * dy;
        int x = x0 + k * dx;
        if (my[y][x] == 1) my_cnt++;
        else if (opp[y][x] == 1) opp_cnt++;
        else                     emp++;
    }
    if (my_cnt == 4) return  WIN_SCORE;
    if (opp_cnt == 4) return -WIN_SCORE;

    int s = 0;
    if (my_cnt == 3 && emp == 1) s += THREE_OPEN;
    if (my_cnt == 2 && emp == 2) s += TWO_OPEN;
    if (opp_cnt == 3 && emp == 1) s -= THREE_OPEN_BLOCK;
    return s;
}

// ── 평가 함수 ────────────────────────────────────────────────
static inline int eval(const State& s) {
    if (s.isDone()) {
        switch (s.getWinningStatus()) {
        case WinningStatus::LOSE: return -WIN_SCORE;
        case WinningStatus::WIN:  return  WIN_SCORE;
        default:                  return  0;
        }
    }

    const int(*my)[W] = s.getMyBoard();
    const int(*opp)[W] = s.getEnemyBoard();
    int score = 0;

    for (int y = 0; y < H; ++y)
        for (int x = 0; x <= W - 4; ++x)
            score += score_window_direct(my, opp, y, x, 0, 1);

    for (int y = 0; y <= H - 4; ++y)
        for (int x = 0; x < W; ++x)
            score += score_window_direct(my, opp, y, x, 1, 0);

    for (int y = 0; y <= H - 4; ++y)
        for (int x = 0; x <= W - 4; ++x)
            score += score_window_direct(my, opp, y, x, 1, 1);

    for (int y = 3; y < H; ++y)
        for (int x = 0; x <= W - 4; ++x)
            score += score_window_direct(my, opp, y, x, -1, 1);

    const int cx = W / 2;
    for (int y = 0; y < H; ++y)
        if (my[y][cx] == 1) score += CENTER_BONUS_PIECE;

    return score;
}

// ── 무브 오더링 ──────────────────────────────────────────────
static inline void order_actions(std::vector<int>& acts) {
    const int c = W / 2;
    std::sort(acts.begin(), acts.end(),
        [c](int a, int b) { return std::abs(a - c) < std::abs(b - c); });
}

// eval 정렬 + 킬러 수를 eval 계산 없이 맨 앞으로 배치
//
// 우선순위:
//   1) killers[0]  — 가장 최근 베타 컷 수
//   2) killers[1]  — 그 이전 베타 컷 수
//   3) 나머지를 eval 내림차순 정렬
//
static inline void order_actions_with_killers(
    const State& state,
    std::vector<int>& acts,
    const KillerTable& kt,
    int ply)
{
    const auto& killers = kt[ply];

    struct MoveScore { int action, score; };
    std::vector<MoveScore> scored;
    scored.reserve(acts.size());

    for (int a : acts) {
        int score;
        if (a == killers[0]) score = INF - 1; // eval 불필요, 최상위 우선
        else if (a == killers[1]) score = INF - 2;
        else {
            State child = state;
            child.advance(a);
            score = -eval(child);
        }
        scored.push_back({ a, score });
    }

    std::sort(scored.begin(), scored.end(),
        [](const MoveScore& l, const MoveScore& r) { return l.score > r.score; });

    for (int i = 0; i < (int)acts.size(); ++i)
        acts[i] = scored[i].action;
}

// ── 알파-베타 네가맥스 (킬러 휴리스틱 적용) ──────────────────
//
// depth : 남은 탐색 깊이
// ply   : 루트로부터 현재 노드까지의 층 (킬러 테이블 인덱스)
//         루트 = 0, 한 수 내려갈 때마다 +1
//
int negamax_alpha_beta(
    State state, int depth, int alpha, int beta,
    TimeKeeper& tk, KillerTable& kt, int ply)
{
    if (depth == 0 || state.isDone()) return eval(state);
    if (tk.isTimeOver())              return eval(state);

    auto acts = state.legalActions();
    if (acts.empty()) return eval(state);

    order_actions_with_killers(state, acts, kt, ply);

    int best = -INF;
    for (int a : acts) {
        State child = state;
        child.advance(a);
        int v = -negamax_alpha_beta(child, depth - 1, -beta, -alpha,
            tk, kt, ply + 1);
        if (v > best)  best = v;
        if (v > alpha) alpha = v;

        if (alpha >= beta) {
            // 베타 컷 발생 → 이 수를 현재 층의 킬러로 등록
            store_killer(kt, ply, a);
            break;
        }
    }
    return best;
}

int alphaBetaAction(const State& state, int max_depth, int time_limit_ms)
{
    auto start = std::chrono::high_resolution_clock::now();
    TimeKeeper tk(time_limit_ms);

    // 킬러 테이블을 탐색 전체에서 공유 (-1 = 비어 있음)
    KillerTable kt;
    for (auto& row : kt) row.fill(-1);

    int bestMove = -1;
    int depth = 1;

    while (depth <= max_depth && !tk.isTimeOver()) {
        auto acts = state.legalActions();
        if (acts.empty()) break;

        // 루트(ply=0)도 킬러 오더링 적용
        order_actions_with_killers(state, acts, kt, 0);

        int localBestMove = acts.front();
        int alpha = -INF, beta = INF;

        for (int a : acts) {
            if (tk.isTimeOver()) break;
            State child = state;
            child.advance(a);
            // 루트 자식 노드의 ply = 1
            int v = -negamax_alpha_beta(child, depth - 1, -beta, -alpha,
                tk, kt, 1);
            if (v > alpha) { alpha = v; localBestMove = a; }
        }

        if (!tk.isTimeOver()) bestMove = localBestMove;
        depth++;
    }

    if (bestMove == -1) {
        auto acts = state.legalActions();
        if (!acts.empty()) bestMove = acts.front();
    }

    auto end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(end - start).count();
    //std::cout << "alpha-beta depth: " << depth - 1 << "\n";
    return bestMove;
}

// ── 네가맥스 (알파-베타 없음, 킬러 미적용) ───────────────────
int negamax(State state, int depth, TimeKeeper& tk) {
    if (depth == 0 || state.isDone()) return eval(state);
    if (tk.isTimeOver())              return eval(state);

    auto acts = state.legalActions();
    if (acts.empty()) return eval(state);

    order_actions(acts);

    int best = -INF;
    for (int a : acts) {
        State child = state;
        child.advance(a);
        int v = -negamax(child, depth - 1, tk);
        if (v > best) best = v;
    }
    return best;
}

int negamaxAction(const State& state, int max_depth, int time_limit_ms) {
    auto start = std::chrono::high_resolution_clock::now();
    TimeKeeper tk(time_limit_ms);

    int bestMove = -1;
    int depth = 1;

    while (depth <= max_depth && !tk.isTimeOver()) {
        auto acts = state.legalActions();
        if (acts.empty()) break;

        order_actions(acts);

        int localBestMove = acts.front();
        int localBestV = -INF;

        for (int a : acts) {
            if (tk.isTimeOver()) break;
            State child = state;
            child.advance(a);
            int v = -negamax(child, depth - 1, tk);
            if (v > localBestV) { localBestV = v; localBestMove = a; }
        }

        if (!tk.isTimeOver()) bestMove = localBestMove;
        depth++;
    }

    if (bestMove == -1) {
        auto acts = state.legalActions();
        if (!acts.empty()) bestMove = acts.front();
    }

    auto end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(end - start).count();
    //std::cout << "depth: " << depth - 1 << "\n";
    return bestMove;
}

// ── 학습 데이터 저장 ─────────────────────────────────────────
static std::ofstream data_file;
static std::ofstream train_data_file;
static std::ofstream valid_data_file;

static void write_header(std::ofstream& out) {
    for (int i = 0; i < 42; i++) out << "c" << i << ",";
    out << "score\n";
}

double normalize_score(double s) {
    double x = s / 10000.0;
    if (x > 1.0) x = 1.0;
    if (x < -1.0) x = -1.0;
    return x;
}

void open_data_file(const std::string& filename) {
    data_file.open(filename);
    write_header(data_file);
}

void open_data_files(const std::string& train_filename, const std::string& valid_filename) {
    train_data_file.open(train_filename);
    valid_data_file.open(valid_filename);
    write_header(train_data_file);
    write_header(valid_data_file);
}

void close_data_file() {
    if (data_file.is_open()) data_file.close();
    if (train_data_file.is_open()) train_data_file.close();
    if (valid_data_file.is_open()) valid_data_file.close();
}

void save_sample(const State& s, bool to_validation) {
    std::ofstream* target = nullptr;
    if (train_data_file.is_open() && valid_data_file.is_open()) {
        target = to_validation ? &valid_data_file : &train_data_file;
    }
    else if (data_file.is_open()) {
        target = &data_file;
    }

    if (target == nullptr || !target->is_open()) return;

    const int(*my)[W] = s.getMyBoard();
    const int(*opp)[W] = s.getEnemyBoard();

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int v = (my[y][x] == 1) ? 1 : (opp[y][x] == 1) ? -1 : 0;
            (*target) << v << ",";
        }
    }
    (*target) << normalize_score(eval(s)) << "\n";
}
