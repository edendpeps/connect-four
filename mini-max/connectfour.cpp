#include "ConnectFourState.hpp"
#include <sstream>
#include <utility>
#include <algorithm>

ConnectFourState::ConnectFourState() {}

bool ConnectFourState::isDone() const {
    return winning_status_ != WinningStatus::NONE;
}

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
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            if (my_board_[y][x] || enemy_board_[y][x]) anyStone = true;

    if (!anyStone) {
        actions.push_back((H / 2) * W + (W / 2));
        return actions;
    }

    // 2) 기존 돌 주변 2칸 이내 빈칸만 후보로
    static constexpr int R = 2;
    bool mark[H][W] = {};

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (my_board_[y][x] || enemy_board_[y][x]) {
                for (int dy = -R; dy <= R; ++dy) {
                    for (int dx = -R; dx <= R; ++dx) {
                        int ny = y + dy, nx = x + dx;
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

    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            if (mark[y][x]) actions.push_back(y * W + x);

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
