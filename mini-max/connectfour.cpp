#include "ConnectFourState.hpp"
#include <sstream>
#include <utility>
#include <algorithm>

ConnectFourState::ConnectFourState() {}

bool ConnectFourState::isDone() const {
    return winning_status_ != WinningStatus::NONE;
}
// ===== 33금수용 헬퍼들 (ConnectFourState.cpp) =====
// =======================
// Gomoku black 33-forbidden (only)
// put this into ConnectFourState.cpp
// action = y*W + x
// =======================

static inline bool insideYX(int y, int x) {
    return 0 <= y && y < H && 0 <= x && x < W;
}

static inline int getStoneAt(const ConnectFourState& st, int y, int x) {
    // 현재 플레이어(my_board_) 돌을 1, 상대(enemy_board_) 돌을 2, 빈칸 0
    if (st.hasMyStone(y, x)) return 1;
    if (st.hasEnemyStone(y,x)) return 2;
    return 0;
}

// (y,x)에서 (dy,dx) 방향으로 "현재 플레이어 돌(=1)" 연속 개수
static inline int runMy(const ConnectFourState& st, int y, int x, int dy, int dx) {
    int cnt = 0;
    while (insideYX(y, x) && st.hasMyStone(y,x) == 1) {
        ++cnt; y += dy; x += dx;
    }
    return cnt;
}

// strict OPEN-THREE: ".xxx." (연속 3 + 양끝 둘 다 빈칸)
// 여기서 x는 "현재 플레이어(흑)" 기준.
// lineLen은 9~11 정도면 충분.
static bool hasOpenThreeStrict(const ConnectFourState& st, int cy, int cx, int dy, int dx) {
    // 중심 (cy,cx)가 방금 둔 돌이라고 가정하고, 그 주변으로 문자열 구성
    // 범위: -5..+5 (총 11칸) 중 보드 밖은 '2'(벽=막힘)처럼 처리
    // '0' = 빈칸, '1' = 내돌, '2' = 상대/벽
    // 우리는 ".111." 패턴만 카운트한다.

    char s[11];
    int idx0 = 5; // 중심 인덱스
    for (int k = -5; k <= 5; ++k) {
        int y = cy + k * dy;
        int x = cx + k * dx;
        int v = 2; // 기본: 벽(막힘)
        if (insideYX(y, x)) v = getStoneAt(st, y, x);
        s[k + 5] = char('0' + v);
    }

    // ".111." == "0 1 1 1 0"
    // 단, 중심(idx0)
    //  그 3개 안에 포함되는 경우만 인정
    for (int i = 0; i + 4 < 11; ++i) {
        if (s[i] == '0' && s[i + 1] == '1' && s[i + 2] == '1' && s[i + 3] == '1' && s[i + 4] == '0') {
            // 중심이 i+1..i+3 안에 들어가야 "내가 방금 둔 수로 생긴 열린3"로 친다
            if (i + 1 <= idx0 && idx0 <= i + 3) return true;
        }
    }
    return false;
}

// 흑(현재 플레이어) 33 금수 판정
bool ConnectFourState::isForbidden33(int action) const {
    int y = action / W;
    int x = action % W;
    if (!insideYX(y, x)) return true;
    if (my_board_[y][x] || enemy_board_[y][x]) return true;

    // 가상 착수: swap/turn 변경 없이 "현재 플레이어(my_board_)"에만 놓고 검사
    ConnectFourState tmp = *this;
    tmp.my_board_[y][x] = 1;

    // 4방향
    static const int DY[4] = { 0, 1, 1, 1 };
    static const int DX[4] = { 1, 0, 1,-1 };

    int openThreeCnt = 0;
    for (int d = 0; d < 4; ++d) {
        if (hasOpenThreeStrict(tmp, y, x, DY[d], DX[d])) {
            openThreeCnt++;
        }
    }

    return openThreeCnt >= 2;
}

// =======================
// 그리고 legalActions()에서 금수 필터링을 이렇게 해
// =======================

// 예시: 네 legalActions()가 "근처 2칸 후보" 방식이면, actions push 전에 if로 걸러.
// 아래는 "전체 빈칸 다 후보"일 때 예시. 네 방식에 맞춰서 'push_back' 직전에만 넣으면 됨.

/*
std::vector<int> ConnectFourState::legalActions() const {
    std::vector<int> actions;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (my_board_[y][x] || enemy_board_[y][x]) continue;
            int a = y*W + x;

            // ★ 여기 한 줄이 핵심: 33 금수는 후보에서 제거
            if (isForbidden33(a)) continue;

            actions.push_back(a);
        }
    }
    return actions;
}
*/


void ConnectFourState::advance(int action)
{
    // action -> (y,x)
    int y0 = action / W;
    int x0 = action % W;

    // 이미 찬 칸이면 방어적으로 무시
    if (my_board_[y0][x0] || enemy_board_[y0][x0]) return;

    // 1) 돌 놓기 (현재 플레이어: my_board_)
    my_board_[y0][x0] = 1;

    auto has5 = [&](int dy, int dx) {
        int c = run(my_board_, y0, x0, dy, dx)
            + run(my_board_, y0, x0, -dy, -dx) - 1;
        return c >= WIN_LEN;
        };

    bool win_now =
        has5(1, 0) || has5(0, 1) || has5(1, 1) || has5(1, -1);

    bool board_full = true;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (my_board_[y][x] == 0 && enemy_board_[y][x] == 0) {
                board_full = false;
                break;
            }
        }
        if (!board_full) break;
    }

    // 2) 턴 넘기기 (C++ 원본 컨벤션 유지!)
    std::swap(my_board_, enemy_board_);
    is_first_ = !is_first_;

    // 3) 다음 플레이어 관점에서 승패 기록
    if (win_now) {
        // 방금 둔 사람이 승리 → 지금 플레이어는 패배
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

    // 1) 보드에 돌이 하나도 없으면 중앙만 허용 (오프닝 고정)
    bool anyStone = false;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (my_board_[y][x] || enemy_board_[y][x]) {
                anyStone = true;
                break;
            }
        }
        if (anyStone) break;
    }

    if (!anyStone) {
        int a = (H / 2) * W + (W / 2);
        actions.push_back(a);
        return actions;
    }

    // 2) 기존 돌 주변 R칸 이내 빈칸만 후보로
    static constexpr int R = 2;
    bool mark[H][W] = {};

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (my_board_[y][x] || enemy_board_[y][x]) {
                for (int dy = -R; dy <= R; ++dy) {
                    for (int dx = -R; dx <= R; ++dx) {
                        int ny = y + dy;
                        int nx = x + dx;
                        if (0 <= ny && ny < H && 0 <= nx && nx < W) {
                            if (!my_board_[ny][nx] && !enemy_board_[ny][nx]) {
                                mark[ny][nx] = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // 3) mark된 칸을 actions로 변환하면서 33 금수(흑만) 필터링
    //    - 네 구조에서 my_board_는 "현재 플레이어" 관점
    //    - 보통 흑=선공이라 is_first_==true인 턴이 흑 턴이라고 가정
    const bool blackTurn = (is_first_ == true);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (!mark[y][x]) continue;

            int a = y * W + x;

            // ★ 흑일 때만 33 금수 적용
            if (blackTurn && isForbidden33(a)) continue;

            actions.push_back(a);
        }
    }

    return actions;
}



WinningStatus ConnectFourState::getWinningStatus() const {
    return winning_status_;
}

std::string ConnectFourState::toString() const
{
    std::stringstream ss;
    ss << "is_first:\t" << (is_first_ ? 1 : 0) << "\n";

    // y=0이 위, y=H-1이 아래로 출력 (원본과 동일한 스타일)
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            char c = '.';
            if (my_board_[y][x] == 1) c = (is_first_ ? 'x' : 'o');
            else if (enemy_board_[y][x] == 1) c = (is_first_ ? 'o' : 'x');
            ss << c;
        }
        ss << "\n";
    }
    return ss.str();
}
