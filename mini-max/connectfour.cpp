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
#include "Monte_Carlo.hpp"
#include <set>
#include <limits> // 추가: 입력 유효성 처리에 필요


int monte_win = 0;
int minimax_win = 0;
int game_draw = 0;
int game_limit = 200;
// 시간을 관리하는 클래스
const int time_limit = 600;
const int minimax_depth = 1000000;
const int INF = 100000000;
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
	// 1. 말 놓기
	std::pair<int, int> coordinate(-1, -1);
	for (int y = 0; y < H; ++y) {
		if (my_board_[y][action] == 0 && enemy_board_[y][action] == 0) {
			my_board_[y][action] = 1;
			coordinate = { y, action };
			break;
		}
	}

	int y0 = coordinate.first;
	int x0 = coordinate.second;

	auto has4 = [&](int dy, int dx) {
		int c = run(my_board_, y0, x0, dy, dx)
			+ run(my_board_, y0, x0, -dy, -dx) - 1;
		return c >= 4;
		};

	bool win_now =
		has4(0, 1) || has4(1, 0) || has4(1, 1) || has4(1, -1);

	bool board_full = false;
	{
		auto acts = legalActions();    // 현재 판 기준으로 더 둘 곳 있는지
		board_full = acts.empty();
	}

	// 2. 턴 넘기기 (항상)
	std::swap(my_board_, enemy_board_);
	is_first_ = !is_first_;

	// 3. 이제 state는 "다음에 둘 사람" 관점이다.
	if (win_now) {
		// 방금 둔 사람이 이겼으니까, 지금 state 입장에선 내가 진 것
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
		//cout << "당신의 차례입니다 (열 번호 0~" << (W - 1) << "): ";
		if (!(cin >> col))
		{
			cin.clear();
			cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			//cout << "숫자를 입력하세요.\n";
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
	bool last_move_by_minimax = false; // 직전에 누가 뒀는지
void playGame(bool first_is_minimax)
{
	int turn_count = 0;


	auto state = State();
	//std::cout << state.toString() << "\n";

	while (!state.isDone())
	{
		bool minimax_turn = (turn_count % 2 == 0) == first_is_minimax;

		if (minimax_turn)
		{
			//std::cout << "MiniMax ------------------------------------\n";
			int action = negamaxAction(state, minimax_depth, time_limit);
			//std::cout << "Turn : " << turn_count << "\n";
			//std::cout << "action " << action << "\n";
			state.advance(action);
		}
		else
		{
			//std::cout << "MonteCarlo ---------------------------------\n";
			int action = MontecarloAction(state, INF, time_limit);
			//std::cout << "Turn : " << turn_count << "\n";
			//std::cout << "action " << action << "\n";
			state.advance(action);
		}

		last_move_by_minimax = minimax_turn;
		//std::cout << state.toString() << "\n";
		turn_count++;
	}

	if (state.getWinningStatus() == WinningStatus::DRAW)
	{
		std::cout << "DRAW\n";
		game_draw++;
	}
	else if (state.getWinningStatus() == WinningStatus::LOSE)
	{
		// 직전에 둔 사람이 이김
		std::cout << "winner: " << (last_move_by_minimax ? "MiniMax" : "MonteCarlo") << "\n";
		if (last_move_by_minimax) minimax_win++;
		else monte_win++;
	}
	else if (state.getWinningStatus() == WinningStatus::WIN)
	{
		// 직전에 둔 사람이 짐 -> 상대가 이김
		std::cout << "winner: " << (last_move_by_minimax ? "MonteCarlo" : "MiniMax") << "\n";
		if (last_move_by_minimax) monte_win++;
		else minimax_win++;
		 
	}
}

int main()
{
	for (int i = 0; i < game_limit; i++) {
		//std::cout << "game_count: " << i << "\n";
		bool first_is_minimax = (i % 2 == 0); // 번갈아 선공
		playGame(first_is_minimax);
	}
	std::cout << "MonteCarlo_win: " << monte_win<<"\n";
	std::cout << "MiniMax_win: " << minimax_win<<"\n";
	std::cout << "Draw: " << game_draw;;

	return 0;
}
