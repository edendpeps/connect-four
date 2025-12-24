#pragma once
#include <vector>
#include <string>
#include <chrono>

// 보드 크기 (여기 한 군데만 정의하고, 다른 .cpp들에서는 중복 정의하지 않기)
constexpr int H = 15;
constexpr int W = 15;
constexpr int WIN_LEN = 5;

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

class ConnectFourState
{
private:

	bool is_first_ = true;               // 선공 여부
	int my_board_[H][W] = {};            // 내 말
	int enemy_board_[H][W] = {};         // 상대 말
	WinningStatus winning_status_ = WinningStatus::NONE;
	inline int run(const int board[H][W], int y, int x, int dy, int dx) const 
	{
		int cnt = 0; 
		while (0 <= y && y < H && 0 <= x && x < W && board[y][x] == 1) {
			++cnt; y += dy; x += dx;
		}
		return cnt;
	}
public:
	bool hasMyStone(int y, int x) const {
		return my_board_[y][x] == 1;
	}

	// y,x에 상대 돌이 있는지
	bool hasEnemyStone(int y, int x) const {
		return enemy_board_[y][x] == 1;
	}

	// 빈칸인지
	bool isEmpty(int y, int x) const {
		return my_board_[y][x] == 0 && enemy_board_[y][x] == 0;
	}
	bool isForbidden33(int action) const;

	ConnectFourState();

	// 게임 종료 여부
	bool isDone() const;

	// 지정한 열(action)에 말 떨어뜨리기 + 턴 전환
	void advance(int action);

	// 현재 플레이어가 둘 수 있는 열 목록
	std::vector<int> legalActions() const;

	// 승패 정보
	WinningStatus getWinningStatus() const;

	// 디버그 출력용 보드 문자열
	std::string toString() const;
};
