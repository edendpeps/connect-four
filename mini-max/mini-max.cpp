#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <iostream>
#include <chrono>
#include "ConnectFourState.hpp"
#include "minimax.hpp"

using State = ConnectFourState;

static constexpr int INF = 100000000;
static constexpr int WIN_SCORE = 100000000;

// eval용 가중치(너가 쓰던 감각 그대로 두고, 강제수에서 더 세게 처리함)
static constexpr int OPEN_FOUR = 80000000;
static constexpr int HALF_FOUR = 2000000;
static constexpr int CLOSED_FOUR = 50000;

static constexpr int OPEN_THREE = 200000;
static constexpr int HALF_THREE = 20000;
static constexpr int TWO_OPEN = 500;

static constexpr int BLOCK_OPEN_FOUR = 90000000;
static constexpr int BLOCK_HALF_FOUR = 3000000;
static constexpr int BLOCK_OPEN_THREE = 250000;

double duration;

// -------------------------
// rows 파싱 + 관점 정규화
// -------------------------
static inline bool inside(int y, int x) {
    return 0 <= y && y < H && 0 <= x && x < W;
}

static inline void parseBoard(const State& s,
    std::vector<std::string>& rows,
    bool& is_first_out)
{
    rows.clear();
    std::stringstream ss(s.toString());
    std::string line;

    std::getline(ss, line);
    {
        auto pos = line.find('\t');
        int v = 1;
        if (pos != std::string::npos) v = std::stoi(line.substr(pos + 1));
        is_first_out = (v != 0);
    }

    for (int i = 0; i < H; ++i) {
        std::getline(ss, line);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        rows.push_back(line);
    }
}

// 현재 플레이어가 항상 'x'가 되도록 정규화
static inline void normalizedRows(const State& s, std::vector<std::string>& rows) {
    bool is_first = true;
    parseBoard(s, rows, is_first);

    if (!is_first) {
        for (auto& row : rows) {
            for (auto& ch : row) {
                if (ch == 'x') ch = 'o';
                else if (ch == 'o') ch = 'x';
            }
        }
    }
}

static inline char at_cell(const std::vector<std::string>& rows, int y, int x) {
    return rows[y][x];
}

// -------------------------
// rows 위에서 수 시뮬레이션
// -------------------------
static inline std::vector<std::string> placeStone(
    const std::vector<std::string>& rows,
    int action, char stone)
{
    auto nxt = rows;
    int y = action / W;
    int x = action % W;
    nxt[y][x] = stone;
    return nxt;
}

// -------------------------
// 5목 / 열린4 판정
// -------------------------
static const int DY[4] = { 1, 0, 1, 1 };
static const int DX[4] = { 0, 1, 1, -1 };

static inline bool hasFive(
    const std::vector<std::string>& rows, char stone)
{
    const int L = WIN_LEN;

    // 가로/세로/대각/역대각 전부 스캔
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int d = 0; d < 4; ++d) {
                int ey = y + DY[d] * (L - 1);
                int ex = x + DX[d] * (L - 1);
                if (!inside(ey, ex)) continue;

                bool ok = true;
                for (int k = 0; k < L; ++k) {
                    if (rows[y + DY[d] * k][x + DX[d] * k] != stone) {
                        ok = false; break;
                    }
                }
                if (ok) return true;
            }
        }
    }
    return false;
}

static inline bool hasOpenFour(
    const std::vector<std::string>& rows, char stone)
{
    // .ssss. 형태가 있으면 열린4
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int d = 0; d < 4; ++d) {
                int ey = y + DY[d] * 3;
                int ex = x + DX[d] * 3;
                if (!inside(ey, ex)) continue;

                bool four = true;
                for (int k = 0; k < 4; ++k) {
                    if (rows[y + DY[d] * k][x + DX[d] * k] != stone) {
                        four = false; break;
                    }
                }
                if (!four) continue;

                int by = y - DY[d], bx = x - DX[d];
                int ny = ey + DY[d], nx = ex + DX[d];

                bool openL = inside(by, bx) && rows[by][bx] == '.';
                bool openR = inside(ny, nx) && rows[ny][nx] == '.';

                if (openL && openR) return true;
            }
        }
    }
    return false;
}

// 열린4의 양끝 빈칸을 모두 수집 (막기용)
static inline std::vector<int> openFourEnds(
    const std::vector<std::string>& rows, char stone)
{
    std::vector<int> ends;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int d = 0; d < 4; ++d) {
                int ey = y + DY[d] * 3;
                int ex = x + DX[d] * 3;
                if (!inside(ey, ex)) continue;

                bool four = true;
                for (int k = 0; k < 4; ++k) {
                    if (rows[y + DY[d] * k][x + DX[d] * k] != stone) {
                        four = false; break;
                    }
                }
                if (!four) continue;

                int by = y - DY[d], bx = x - DX[d];
                int ny = ey + DY[d], nx = ex + DX[d];

                if (inside(by, bx) && rows[by][bx] == '.') {
                    ends.push_back(by * W + bx);
                }
                if (inside(ny, nx) && rows[ny][nx] == '.') {
                    ends.push_back(ny * W + nx);
                }
            }
        }
    }

    // 중복 제거
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return ends;
}

// -------------------------
// 후보 수 생성(오목 전용)
//  - legalActions만 쓰면 상대 위협 놓칠 수 있어서
//    rows에서 radius 2 후보 다시 만듦
// -------------------------
static inline std::vector<int> genCandidates(const std::vector<std::string>& rows)
{
    const int R = 2;
    bool mark[H][W] = {};
    bool anyStone = false;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (rows[y][x] != '.') {
                anyStone = true;
                for (int dy = -R; dy <= R; ++dy) {
                    for (int dx = -R; dx <= R; ++dx) {
                        int ny = y + dy, nx = x + dx;
                        if (inside(ny, nx) && rows[ny][nx] == '.') {
                            mark[ny][nx] = true;
                        }
                    }
                }
            }
        }
    }

    std::vector<int> acts;
    if (!anyStone) {
        acts.push_back((H / 2) * W + (W / 2));
        return acts;
    }

    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            if (mark[y][x]) acts.push_back(y * W + x);
    return acts;
}



// -------------------------
// eval (패턴식 유지)
// -------------------------
static inline int score_window(
    const std::vector<std::string>& rows,
    int y0, int x0, int dy, int dx, int L)
{
    int my = 0, opp = 0, emp = 0;
    for (int k = 0; k < L; k++) {
        char c = at_cell(rows, y0 + k * dy, x0 + k * dx);
        if (c == 'x') my++;
        else if (c == 'o') opp++;
        else emp++;
    }

    if (my == L)  return  WIN_SCORE;
    if (opp == L) return -WIN_SCORE;

    int by = y0 - dy, bx = x0 - dx;
    int ey = y0 + L * dy, ex = x0 + L * dx;

    bool openL = inside(by, bx) && at_cell(rows, by, bx) == '.';
    bool openR = inside(ey, ex) && at_cell(rows, ey, ex) == '.';

    int s = 0;

    if (my == 4 && emp == 1) {
        if (openL && openR) s += OPEN_FOUR;
        else if (openL || openR) s += HALF_FOUR;
        else s += CLOSED_FOUR;
    }
    else if (my == 3 && emp == 2) {
        if (openL && openR) s += OPEN_THREE;
        else if (openL || openR) s += HALF_THREE;
    }
    else if (my == 2 && emp == 3) {
        if (openL && openR) s += TWO_OPEN;
    }

    if (opp == 4 && emp == 1) {
        if (openL && openR) s -= BLOCK_OPEN_FOUR;
        else if (openL || openR) s -= BLOCK_HALF_FOUR;
        else s -= CLOSED_FOUR;
    }
    else if (opp == 3 && emp == 2) {
        if (openL && openR) s -= BLOCK_OPEN_THREE;
        else if (openL || openR) s -= HALF_THREE;
    }

    return s;
}

static inline int eval(const State& st)
{
    if (st.isDone()) {
        switch (st.getWinningStatus()) {
        case WinningStatus::LOSE: return -WIN_SCORE;
        case WinningStatus::WIN:  return  WIN_SCORE;
        default: return 0;
        }
    }

    std::vector<std::string> rows;
    normalizedRows(st, rows);

    int score = 0;
    const int L = WIN_LEN;

    for (int y = 0; y < H; ++y)
        for (int x = 0; x <= W - L; ++x)
            score += score_window(rows, y, x, 0, 1, L);

    for (int y = 0; y <= H - L; ++y)
        for (int x = 0; x < W; ++x)
            score += score_window(rows, y, x, 1, 0, L);

    for (int y = 0; y <= H - L; ++y)
        for (int x = 0; x <= W - L; ++x)
            score += score_window(rows, y, x, 1, 1, L);

    for (int y = L - 1; y < H; ++y)
        for (int x = 0; x <= W - L; ++x)
            score += score_window(rows, y, x, -1, 1, L);

    return score;
}

// -------------------------
// 행동 정렬(중앙 근접)
// -------------------------
static inline void order_actions(std::vector<int>& acts) {
    const int cy = H / 2, cx = W / 2;
    std::sort(acts.begin(), acts.end(),
        [=](int a, int b) {
            int ay = a / W, ax = a % W;
            int by = b / W, bx = b % W;
            int da = std::abs(ay - cy) + std::abs(ax - cx);
            int db = std::abs(by - cy) + std::abs(bx - cx);
            return da < db;
        });
}
static long long nodes = 0;
static int max_reached = 0;

// -------------------------
// 네가맥스 (알파베타 없음)
// -------------------------
int negamax(State state, int depth, TimeKeeper& tk, int ply_from_root)
{
    nodes++;
    if (ply_from_root >= max_reached) max_reached = ply_from_root;

    if (depth == 0 || state.isDone()) return eval(state);
    if (tk.isTimeOver()) return eval(state);

    auto acts = state.legalActions();
    if (acts.empty()) return eval(state);

    order_actions(acts);

    int best = -INF;
    for (int a : acts) {
        State child = state;
        child.advance(a);
        int v = -negamax(child, depth - 1, tk, ply_from_root + 1);
        if (v > best) best = v;
    }
    return best;
}

int negamaxAction(const State& state, int max_depth, int time_limit_ms)
{
    // -------- 여기부터 일반 네가맥스 --------
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

            int v = -negamax(child, depth - 1, tk,1 );
            if (v > localBestV) {
                localBestV = v;
                localBestMove = a;
            }
        }

        // 이 깊이는 어느 정도 봤으니, 일단 이걸 “최신 최선 수”로 저장
        if (!tk.isTimeOver()) {
            bestMove = localBestMove;
        }

        depth++;
    }

    if (bestMove == -1) {
        // 시간 시작부터 거의 0ms였던 극단 케이스 방어
        auto acts = state.legalActions();
        if (!acts.empty()) bestMove = acts.front();
    }
    auto end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "depth: " << depth - 1 << "\n";
    std::cout << "duration: " << duration << "\n";
    return bestMove;

}
