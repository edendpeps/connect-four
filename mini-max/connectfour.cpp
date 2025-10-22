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
#pragma GCC diagnostic ignored "-Wsign-compare"
std::random_device rnd;
std::mt19937 mt_for_action(0);

// �ð��� �����ϴ� Ŭ����
class TimeKeeper
{
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    int64_t time_threshold_;

public:
    // �ð� ������ �и��� ������ �����ؼ� �ν��Ͻ��� �����Ѵ�.
    TimeKeeper(const int64_t& time_threshold)
        : start_time_(std::chrono::high_resolution_clock::now()),
        time_threshold_(time_threshold)
    {
    }

    // �ν��Ͻ��� ������ �������� ������ �ð� ������ �ʰ����� �ʾҴ��� �����Ѵ�.
    bool isTimeOver() const
    {
        auto diff = std::chrono::high_resolution_clock::now() - this->start_time_;
        return std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() >= time_threshold_;
    }
};

constexpr const int H = 6; // �̷��� ����
constexpr const int W = 7; // �̷��� �ʺ�

using ScoreType = int64_t;
constexpr const ScoreType INF = 1000000000LL;

enum WinningStatus
{
    WIN,
    LOSE,
    DRAW,
    NONE,
};

class ConnectFourState
{
private:
    static constexpr const int dx[2] = { 1, -1 };          // �̵� ������ x����
    static constexpr const int dy_right_up[2] = { 1, -1 }; // "��"�밢�� ������ x����
    static constexpr const int dy_left_up[2] = { -1, 1 };  // "\" �밢�� ������ x����

    bool is_first_ = true; // ���� ����
    int my_board_[H][W] = {};
    int enemy_board_[H][W] = {};
    WinningStatus winning_status_ = WinningStatus::NONE;

public:
    ConnectFourState()
    {
    }

    // [��� ���ӿ��� ����] : ���� ���� ����
    bool isDone() const
    {
        return winning_status_ != WinningStatus::NONE;
    }

    // [��� ���ӿ��� ����] : ������ action���� ������ 1�� �����ϰ� ���� �÷��̾� ������ ���������� �����.
    void advance(const int action)
    {
        std::pair<int, int> coordinate;
        for (int y = 0; y < H; y++)
        {
            if (this->my_board_[y][action] == 0 && this->enemy_board_[y][action] == 0)
            {
                this->my_board_[y][action] = 1;
                coordinate = std::pair<int, int>(y, action);
                break;
            }
        }

        { // ���� �������� �����ΰ� �����Ѵ�.

            auto que = std::deque<std::pair<int, int>>();
            que.emplace_back(coordinate);
            std::vector<std::vector<bool>> check(H, std::vector<bool>(W, false));
            int count = 0;
            while (!que.empty())
            {
                const auto& tmp_cod = que.front();
                que.pop_front();
                ++count;
                if (count >= 4)
                {
                    this->winning_status_ = WinningStatus::LOSE; // �ڽ��� ���� �����̸� ������ �й�
                    break;
                }
                check[tmp_cod.first][tmp_cod.second] = true;

                for (int action = 0; action < 2; action++)
                {
                    int ty = tmp_cod.first;
                    int tx = tmp_cod.second + dx[action];

                    if (ty >= 0 && ty < H && tx >= 0 && tx < W && my_board_[ty][tx] == 1 && !check[ty][tx])
                    {
                        que.emplace_back(ty, tx);
                    }
                }
            }
        }
        if (!isDone())
        { // "��"�������� �����ΰ� �����Ѵ�.
            auto que = std::deque<std::pair<int, int>>();
            que.emplace_back(coordinate);
            std::vector<std::vector<bool>> check(H, std::vector<bool>(W, false));
            int count = 0;
            while (!que.empty())
            {
                const auto& tmp_cod = que.front();
                que.pop_front();
                ++count;
                if (count >= 4)
                {
                    this->winning_status_ = WinningStatus::LOSE; // �ڽ��� ���� �����̸� ������ �й�
                    break;
                }
                check[tmp_cod.first][tmp_cod.second] = true;

                for (int action = 0; action < 2; action++)
                {
                    int ty = tmp_cod.first + dy_right_up[action];
                    int tx = tmp_cod.second + dx[action];

                    if (ty >= 0 && ty < H && tx >= 0 && tx < W && my_board_[ty][tx] == 1 && !check[ty][tx])
                    {
                        que.emplace_back(ty, tx);
                    }
                }
            }
        }

        if (!isDone())
        { // "\"�������� �����ΰ� �����Ѵ�.

            auto que = std::deque<std::pair<int, int>>();
            que.emplace_back(coordinate);
            std::vector<std::vector<bool>> check(H, std::vector<bool>(W, false));
            int count = 0;
            while (!que.empty())
            {
                const auto& tmp_cod = que.front();
                que.pop_front();
                ++count;
                if (count >= 4)
                {
                    this->winning_status_ = WinningStatus::LOSE; // �ڽ��� ���� �����̸� ������ �й�
                    break;
                }
                check[tmp_cod.first][tmp_cod.second] = true;

                for (int action = 0; action < 2; action++)
                {
                    int ty = tmp_cod.first + dy_left_up[action];
                    int tx = tmp_cod.second + dx[action];

                    if (ty >= 0 && ty < H && tx >= 0 && tx < W && my_board_[ty][tx] == 1 && !check[ty][tx])
                    {
                        que.emplace_back(ty, tx);
                    }
                }
            }
        }
        if (!isDone())
        { // ���� �������� �����ΰ� �����Ѵ�.

            int ty = coordinate.first;
            int tx = coordinate.second;
            bool is_win = true;
            for (int i = 0; i < 4; i++)
            {
                bool is_mine = (ty >= 0 && ty < H && tx >= 0 && tx < W && my_board_[ty][tx] == 1);

                if (!is_mine)
                {
                    is_win = false;
                    break;
                }
                --ty;
            }
            if (is_win)
            {
                this->winning_status_ = WinningStatus::LOSE; // �ڽ��� ���� �����̸� ������ �й�
            }
        }

        std::swap(my_board_, enemy_board_);
        is_first_ = !is_first_;
        if (this->winning_status_ == WinningStatus::NONE && legalActions().size() == 0)
        {
            this->winning_status_ = WinningStatus::DRAW;
        }
    }

    // [��� ���ӿ��� ����] : ���� �÷��̾ ������ �ൿ�� ��� ȹ���Ѵ�.
    std::vector<int> legalActions() const
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

    // [��� ���ӿ��� ����] : ���� ������ ȹ���Ѵ�.
    WinningStatus getWinningStatus() const
    {
        return this->winning_status_;
    }

    // [�ʼ��� �ƴ����� �����ϸ� ��] : ���� ���� ��Ȳ�� ���ڿ��� �����.
    std::string toString() const
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
};

using State = ConnectFourState;

using AIFunction = std::function<int(const State&)>;
using StringAIPair = std::pair<std::string, AIFunction>;

// �������� �ൿ�� �����Ѵ�.
int randomAction(const State& state)
{
    auto legal_actions = state.legalActions();
    return legal_actions[mt_for_action() % (legal_actions.size())];
}

// ������ 1ȸ �÷����ؼ� ���� ��Ȳ�� ǥ���Ѵ�.
void playGame()
{
    using std::cout;
    using std::endl;
    auto state = State();
    cout << state.toString() << endl;
    while (!state.isDone())
    {
        // 1p
        {
            cout << "1p ------------------------------------" << endl;
            int action = randomAction(state);
            cout << "action " << action << endl;
            state.advance(action); // (a-1) ���⼭ ������ �ٲ� 2p ������ �ȴ�.
            cout << state.toString() << endl;
            if (state.isDone())
            {

                switch (state.getWinningStatus()) // (a-2) a-1���� 2p ������ �Ǿ����Ƿ� WIN�̶�� 2p�� �¸�
                {
                case (WinningStatus::WIN):
                    cout << "winner: "
                        << "2p" << endl;
                    break;
                case (WinningStatus::LOSE):
                    cout << "winner: "
                        << "1p" << endl;
                    break;
                default:
                    cout << "DRAW" << endl;
                    break;
                }
                break;
            }
        }
        // 2p
        {
            cout << "2p ------------------------------------" << endl;
            int action = randomAction(state);
            cout << "action " << action << endl;
            state.advance(action); // (b-1) ���⼭ ������ �ٲ� 1p ������ �ȴ�.
            cout << state.toString() << endl;
            if (state.isDone())
            {

                switch (state.getWinningStatus()) // (b-2) b-1���� 1p ������ �Ǿ����Ƿ� WIN�̶�� 1p�� �¸�
                {
                case (WinningStatus::WIN):
                    cout << "winner: "
                        << "1p" << endl;
                    break;
                case (WinningStatus::LOSE):
                    cout << "winner: "
                        << "2p" << endl;
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