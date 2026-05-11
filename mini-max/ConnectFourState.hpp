#pragma once
#include <vector>
#include <string>
#include <chrono>

constexpr int H = 6;
constexpr int W = 7;

enum class WinningStatus {
    WIN,
    LOSE,
    DRAW,
    NONE,
};

class TimeKeeper
{
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    int64_t time_threshold_;

public:
    TimeKeeper(const int64_t& time_threshold)
        : start_time_(std::chrono::high_resolution_clock::now()),
        time_threshold_(time_threshold)
    {
    }

    bool isTimeOver() const
    {
        auto diff = std::chrono::high_resolution_clock::now() - this->start_time_;
        return std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() >= time_threshold_;
    }
};

class ConnectFourState
{
private:
    static constexpr int dx[2] = { 1, -1 };
    static constexpr int dy_right_up[2] = { 1, -1 };
    static constexpr int dy_left_up[2] = { -1, 1 };

    bool is_first_ = true;
    int my_board_[H][W] = {};
    int enemy_board_[H][W] = {};
    WinningStatus winning_status_ = WinningStatus::NONE;

public:
    ConnectFourState();

    bool isDone() const;
    void advance(int action);
    std::vector<int> legalActions() const;
    WinningStatus getWinningStatus() const;
    std::string toString() const;

    // ── 직접 보드 접근자 (파싱 제거용) ──────────────────────────
    // my_board_[y][x]  : 현재 차례 플레이어의 돌 (1 = 있음, 0 = 없음)
    // enemy_board_[y][x]: 상대 플레이어의 돌
    // y=0 이 바닥, y=H-1 이 천장 (advance()와 동일 좌표계)
    const int(*getMyBoard()    const)[W] { return my_board_; }
    const int(*getEnemyBoard() const)[W] { return enemy_board_; }
    bool isFirst() const { return is_first_; }
};