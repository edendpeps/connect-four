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
#include "ConnectFourState.hpp"
#include "minimax.hpp"
#include <set>
#include <limits> // 추가: 입력 유효성 처리에 필요
std::random_device rnd;
std::mt19937 mt_for_action(0);

// 시간을 관리하는 클래스
class TimeKeeper
{
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    int64_t time_threshold_;

public:
    // 시간 제한을 밀리초 단위로 지정해서 인스턴스를 생성한다.
    TimeKeeper(const int64_t& time_threshold)
        : start_time_(std::chrono::high_resolution_clock::now()),
        time_threshold_(time_threshold)
    {
    }

    // 인스턴스를 생성한 시점부터 지정한 시간 제한을 초과하지 않았는지 판정한다.
    bool isTimeOver() const
    {
        auto diff = std::chrono::high_resolution_clock::now() - this->start_time_;
        return std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() >= time_threshold_;
    }
};

ConnectFourState::ConnectFourState() {}

bool ConnectFourState::isDone() const {
    return winning_status_ != WinningStatus::NONE;
}

// helper: (y,x)에서 (dy,dx) 방향으로 내 돌 연속 길이
inline int run(const int board[H][W], int y, int x, int dy, int dx) {
    int cnt = 0;
    while (y >= 0 && y < H && x >= 0 && x < W && board[y][x] == 1) {
        ++cnt; y += dy; x += dx;
    }
    return cnt;
}
void ConnectFourState::advance(const int action)
{
    // 말 놓기
    std::pair<int, int> coordinate(-1, -1);
    for (int y = 0; y < H; ++y) {
        if (my_board_[y][action] == 0 && enemy_board_[y][action] == 0) {
            my_board_[y][action] = 1;
            coordinate = { y, action };
            break;
        }
    }
    // 안전장치
    // assert(coordinate.first != -1);

    int y0 = coordinate.first;
    int x0 = coordinate.second;

    auto has4 = [&](int dy, int dx) {
        // 양방향 합산에서 중앙을 두 번 세니까 -1
        int c = run(my_board_, y0, x0, dy, dx)
            + run(my_board_, y0, x0, -dy, -dx) - 1;
        return c >= 4;
        };

    // 4방향 체크: 가로, 세로, 대각(\,/)
    if (has4(0, 1) || has4(1, 0) || has4(1, 1) || has4(1, -1)) {
        // "방금 둔 쪽 승리 → LOSE" (네 코딩 규칙 유지)
        this->winning_status_ = WinningStatus::LOSE;
        return; // ★ 승리 시 swap하지 말고 바로 종료
    }

    // 더 둘 곳 없으면 무승부
    if (legalActions().empty()) {
        this->winning_status_ = WinningStatus::DRAW;
        return;
    }

    // 턴 전환
    std::swap(my_board_, enemy_board_);
    is_first_ = !is_first_;
}

std::vector<int> ConnectFourState::legalActions() const
{
    std::vector<int> actions;
    for (int x = 0; x < W; x++)
        for (int y = H - 1; y >= 0; y--)
        {
            if (my_board_[y][x] == 0 && enemy_board_[y][x] == 0)
            {
                actions.emplace_back(x);
                break;
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

using AIFunction = std::function<int(const State&)>;
using StringAIPair = std::pair<std::string, AIFunction>;

// 무작위 행동
int randomAction(const State& state)
{
    auto legal_actions = state.legalActions();
    return legal_actions[mt_for_action() % (legal_actions.size())];
}

// 사람 입력(1P): 열 번호를 입력받아 검증
int humanAction(const State& state)
{
    using std::cout;
    using std::cin;
    using std::endl;

    auto legal = state.legalActions();
    std::set<int> ok(legal.begin(), legal.end());

    int col;
    while (true)
    {
        cout << "당신의 차례입니다 (열 번호 0~" << (W - 1) << "): ";
        if (!(cin >> col))
        {
            cin.clear();
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            cout << "숫자를 입력하세요.\n";
            continue;
        }
        if (0 <= col && col < W && ok.count(col))
        {
            return col;
        }
        cout << "그 열은 둘 수 없습니다. 가능한 열: ";
        for (auto c : legal) cout << c << " ";
        cout << endl;
    }
}

// 게임을 1회 플레이: 1P(사람), 2P(랜덤 AI)
void playGame()
{
    using std::cout;
    using std::endl;
    auto state = State();
    cout << state.toString() << endl;
    while (!state.isDone())
    {
        // 1p (사람)
        {
            cout << "1p ------------------------------------" << endl;
            int action = humanAction(state);
            cout << "action " << action << endl;
            state.advance(action); // 여기서 시점이 바뀌어서 2p 시점이 된다.
            cout << state.toString() << endl;
            if (state.isDone())
            {
                switch (state.getWinningStatus()) // 여기서 WIN은 2p 승
                {
                case (WinningStatus::WIN):
                    cout << "winner: 2p" << endl;
                    break;
                case (WinningStatus::LOSE):
                    cout << "winner: 1p" << endl;
                    break;
                default:
                    cout << "DRAW" << endl;
                    break;
                }
                break;
            }
        }
        // 2p (랜덤 AI)
        {
            cout << "2p ------------------------------------" << endl;
            int action = negamaxAction(state, 6);
            std::cout << duration << "ms\n";
            cout << "action " << action << endl;
            state.advance(action); // 여기서 시점이 바뀌어서 1p 시점이 된다.
            cout << state.toString() << endl;
            if (state.isDone())
            {
                switch (state.getWinningStatus()) // 여기서 WIN은 1p 승
                {
                case (WinningStatus::WIN):
                    cout << "winner: 1p" << endl;
                    break;
                case (WinningStatus::LOSE):
                    cout << "winner: 2p" << endl;
                    break;
                default:
                    cout << "DRAW" << endl;
                    break;
                }
                break;
            }
        }
    }
}

int main()
{
    playGame();
    return 0;
}
