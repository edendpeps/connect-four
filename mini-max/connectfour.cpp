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
#include <set>
#include <limits> // �߰�: �Է� ��ȿ�� ó���� �ʿ�
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

ConnectFourState::ConnectFourState() {}

bool ConnectFourState::isDone() const {
    return winning_status_ != WinningStatus::NONE;
}


void ConnectFourState::advance(const int action)
{
    std::pair<int, int> coordinate(-1, -1);
    for (int y = 0; y < H; y++)
    {
        if (this->my_board_[y][action] == 0 && this->enemy_board_[y][action] == 0)
        {
            this->my_board_[y][action] = 1;
            coordinate = std::pair<int, int>(y, action);
            break;
        }
    }

    // ���� üũ
    {
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
                this->winning_status_ = WinningStatus::LOSE; // �ڽ��� ���� 4�����̸� ��� �й�
                break;
            }
            check[tmp_cod.first][tmp_cod.second] = true;

            for (int a = 0; a < 2; a++)
            {
                int ty = tmp_cod.first;
                int tx = tmp_cod.second + dx[a];

                if (ty >= 0 && ty < H && tx >= 0 && tx < W && my_board_[ty][tx] == 1 && !check[ty][tx])
                {
                    que.emplace_back(ty, tx);
                }
            }
        }
    }
    // "��" üũ
    if (!isDone())
    {
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
                this->winning_status_ = WinningStatus::LOSE;
                break;
            }
            check[tmp_cod.first][tmp_cod.second] = true;

            for (int a = 0; a < 2; a++)
            {
                int ty = tmp_cod.first + dy_right_up[a];
                int tx = tmp_cod.second + dx[a];

                if (ty >= 0 && ty < H && tx >= 0 && tx < W && my_board_[ty][tx] == 1 && !check[ty][tx])
                {
                    que.emplace_back(ty, tx);
                }
            }
        }
    }
    // "\" üũ
    if (!isDone())
    {
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
                this->winning_status_ = WinningStatus::LOSE;
                break;
            }
            check[tmp_cod.first][tmp_cod.second] = true;

            for (int a = 0; a < 2; a++)
            {
                int ty = tmp_cod.first + dy_left_up[a];
                int tx = tmp_cod.second + dx[a];

                if (ty >= 0 && ty < H && tx >= 0 && tx < W && my_board_[ty][tx] == 1 && !check[ty][tx])
                {
                    que.emplace_back(ty, tx);
                }
            }
        }
    }
    // ���� üũ
    if (!isDone())
    {
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
            this->winning_status_ = WinningStatus::LOSE; // �ڽ��� ���� 4�����̸� ��� �й�
        }
    }

    std::swap(my_board_, enemy_board_);
    is_first_ = !is_first_;
    if (this->winning_status_ == WinningStatus::NONE && legalActions().size() == 0)
    {
        this->winning_status_ = WinningStatus::DRAW;
    }
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

// ������ �ൿ
int randomAction(const State& state)
{
    auto legal_actions = state.legalActions();
    return legal_actions[mt_for_action() % (legal_actions.size())];
}

// ��� �Է�(1P): �� ��ȣ�� �Է¹޾� ����
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
        cout << "����� �����Դϴ� (�� ��ȣ 0~" << (W - 1) << "): ";
        if (!(cin >> col))
        {
            cin.clear();
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            cout << "���ڸ� �Է��ϼ���.\n";
            continue;
        }
        if (0 <= col && col < W && ok.count(col))
        {
            return col;
        }
        cout << "�� ���� �� �� �����ϴ�. ������ ��: ";
        for (auto c : legal) cout << c << " ";
        cout << endl;
    }
}

// ������ 1ȸ �÷���: 1P(���), 2P(���� AI)
void playGame()
{
    using std::cout;
    using std::endl;
    auto state = State();
    cout << state.toString() << endl;
    while (!state.isDone())
    {
        // 1p (���)
        {
            cout << "1p ------------------------------------" << endl;
            int action = humanAction(state);
            cout << "action " << action << endl;
            state.advance(action); // ���⼭ ������ �ٲ� 2p ������ �ȴ�.
            cout << state.toString() << endl;
            if (state.isDone())
            {
                switch (state.getWinningStatus()) // ���⼭ WIN�� 2p ��
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
        // 2p (���� AI)
        {
            cout << "2p ------------------------------------" << endl;
            int action = randomAction(state);
            cout << "action " << action << endl;
            state.advance(action); // ���⼭ ������ �ٲ� 1p ������ �ȴ�.
            cout << state.toString() << endl;
            if (state.isDone())
            {
                switch (state.getWinningStatus()) // ���⼭ WIN�� 1p ��
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
