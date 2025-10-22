#pragma once
#include <vector>
#include <string>

// ���� ũ�� (���� �� ������ �����ϰ�, �ٸ� .cpp�鿡���� �ߺ� �������� �ʱ�)
constexpr int H = 6;
constexpr int W = 7;

enum class WinningStatus {
    WIN,
    LOSE,
    DRAW,
    NONE,
};

class ConnectFourState
{
private:
    static constexpr int dx[2] = { 1, -1 };          // ���� ����
    static constexpr int dy_right_up[2] = { 1, -1 }; // "��" �밢
    static constexpr int dy_left_up[2] = { -1, 1 };  // "\" �밢

    bool is_first_ = true;               // ���� ����
    int my_board_[H][W] = {};            // �� ��
    int enemy_board_[H][W] = {};         // ��� ��
    WinningStatus winning_status_ = WinningStatus::NONE;

public:
    ConnectFourState();

    // ���� ���� ����
    bool isDone() const;

    // ������ ��(action)�� �� ����߸��� + �� ��ȯ
    void advance(int action);

    // ���� �÷��̾ �� �� �ִ� �� ���
    std::vector<int> legalActions() const;

    // ���� ����
    WinningStatus getWinningStatus() const;

    // ����� ��¿� ���� ���ڿ�
    std::string toString() const;
};
