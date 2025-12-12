#include <string>
#include <array>
#include <vector>
#include <sstream>
#include "raylib.h"
#include <utility>
#include <random>
#include <assert.h>
#include <math.h>
#include<stdio.h>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <functional>
#include <queue>
#include "ConnectFourState.hpp"
#include "minimax.hpp"
#include "Monte_Carlo.hpp"
#include <set>
#include "raylib.h"
#include <limits> // 추가: 입력 유효성 처리에 필요

bool mini_first = true;
static constexpr int CELL = 80;
static constexpr int PAD = 80;
std::string turn;
std::string what_algo;
// 시간을 관리하는 클래스
const int time_limit = 2000;
const int minimax_depth = 6;
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
//int humanAction(const State& state)
//{
//	using std::cout;
//	using std::cin;
//	using std::endl;
//
//	auto legal = state.legalActions();
//	std::set<int> ok(legal.begin(), legal.end());
//
//	int col;
//	while (true)
//	{
//		cout << "당신의 차례입니다 (열 번호 0~" << (W - 1) << "): ";
//		if (!(cin >> col))
//		{
//			cin.clear();
//			cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
//			cout << "숫자를 입력하세요.\n";
//			continue;
//		}
//		if (0 <= col && col < W && ok.count(col))
//		{
//			return col;
//		}
//		cout << "그 열은 둘 수 없습니다. 가능한 열: ";
//		for (auto c : legal) cout << c << " ";
//		cout << endl;
//	}
//}

// 게임을 1회 플레이: 1P(사람), 2P(랜덤 AI)
void playGame(bool human1p, std::string what_algo)
{

	int turn_count = 0;
	std::string monte = "MonteCarlo";
	std::string minmax = "MiniMax";
	using std::cout;
	using std::endl;
	auto state = State();
	cout << state.toString() << endl;
	while (!state.isDone())
	{
		//if (human1p)
		//{
		//	// 1p (사람)
		//	{
		//		cout << "1p ------------------------------------" << endl;
		//		int action = humanAction(state);
		//		cout << "action " << action << endl;
		//		state.advance(action); // 여기서 시점이 바뀌어서 2p 시점이 된다.
		//		cout << state.toString() << endl;
		//		if (state.isDone())
		//		{
		//			switch (state.getWinningStatus()) // 여기서 WIN은 2p 승
		//			{
		//			case (WinningStatus::WIN):
		//				cout << "winner: 2p" << endl;
		//				break;
		//			case (WinningStatus::LOSE):
		//				cout << "winner: 1p" << endl;
		//				break;
		//			default:
		//				cout << "DRAW" << endl;
		//				break;
		//			}
		//			break;
		//		}
		//	}
		//}

		//1p (Ai)
		if (mini_first)
		{

			{
				cout << minmax << " ------------------------------------" << endl;
				int action = negamaxAction(state, minimax_depth, time_limit);
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
		// 2p (AI)
		if (mini_first)
		{
			{
				turn_count++;
				cout << monte << " ------------------------------------" << endl;
				int action = MontecarloAction(state, INF, time_limit);
				std::cout << "Turn : " << turn_count << endl << duration << "ms\n";
				cout << "action " << action << endl;
				state.advance(action); // 여기서 시점이 바뀌어서 1p 시점이 된다.
				cout << state.toString() << endl;
				if (state.isDone())
				{
					switch (state.getWinningStatus()) // 여기서 WIN은 1p 승
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
		}
	}
}

int main()
{
	/*while (1)
	{

		std::cout << "type 1P or 2P\n";
		std::cin >> turn;
		ConnectFourState state;
		std::transform(turn.begin(), turn.end(), turn.begin(),
			[](unsigned char c) { return std::tolower(c); });
		if (turn == "1p" || turn == "2p")
		{
			break;
		}
		else
		{
			std::cout << "wrong turn input.\n";
		}
	}
	
	bool human1p = (turn == "1p");
	bool humanturn = human1p;
	*/
	const int screenW = W * CELL + PAD * 2;  // W = 7
	const int screenH = H * CELL + PAD * 2;  // H = 6

	InitWindow(screenW, screenH, "Connect Four - Human vs AI");
	SetTargetFPS(60);

	ConnectFourState state;

	// --- 게임 설정 ---
	bool humanIsFirst = false;   // 사람 선공이면 true, 후공이면 false
	bool humanTurn = humanIsFirst;   // 현재 턴이 사람인지 여부

	// AI 설정
	int minimax_depth = 100;
	int time_limit_ms = 2000;
	int playout_num = 100000000; // 몬테카를로용 (INF 정도)

	// 어떤 알고리즘 쓸지 (true = 미니맥스, false = 몬테카를로)
	bool useMinimax = true;

	bool gameOver = false;
	std::string resultText = "";

	while (!WindowShouldClose())
	{
		//// ----------- 입력 (사람 턴) -----------
		//if (!gameOver && humanTurn && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
		//{
		//	int mx = GetMouseX() - PAD;
		//	int my = GetMouseY() - PAD;

		//	// 보드 범위 안인지 체크 (열만 중요)
		//	if (mx >= 0 && my >= 0) {
		//		int x = mx / CELL;  // 열 번호
		//		// int y = my / CELL; // y는 커넥트포에선 안 씀

		//		if (0 <= x && x < W) {
		//			int col = x;

		//			// 현재 둘 수 있는 열인지 확인
		//			auto legal = state.legalActions();  // vector<int> 열 리스트
		//			if (std::find(legal.begin(), legal.end(), col) != legal.end()) {
		//				state.advance(col);   // 실제 말 떨어뜨리기

		//				// 여기서 바로 게임 끝났는지 체크
		//				if (state.isDone()) {
		//					gameOver = true;
		//					// state는 "다음에 둘 사람" 기준이므로
		//					// 방금 둔 사람(= 인간)이 이겼으면 LOSE 로 잡힘
		//					if (state.getWinningStatus() == WinningStatus::LOSE) {
		//						resultText = "You WIN!";
		//					}
		//					else if (state.getWinningStatus() == WinningStatus::WIN) {
		//						resultText = "AI WINS!";
		//					}
		//					else {
		//						resultText = "DRAW";
		//					}
		//				}
		//				else {
		//					humanTurn = false;  // 턴 넘기기
		//				}
		//			}
		//		}
		//	}
		//}
		if (!gameOver && !humanTurn && !IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
		{
			int a;
			a = negamaxAction(state, minimax_depth, time_limit_ms);
			
			state.advance(a);

			if (state.isDone()) {
				gameOver = true;
				// 방금 둔 사람이 AI 이므로
				if (state.getWinningStatus() == WinningStatus::LOSE) {
					resultText = "Montecarlo WINS!";
				}
				else if (state.getWinningStatus() == WinningStatus::WIN) {
					resultText = "Minimax WIN!";
				}
				else {
					resultText = "DRAW";
				}
			}
			else {
				humanTurn = false;  // 다시 사람 차례
			}
		}

		// ----------- AI 턴 -----------
		if (!gameOver && !humanTurn && !IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
		{
			int a;
			a = MontecarloAction(state, playout_num, time_limit_ms);
			

			state.advance(a);

			if (state.isDone()) {
				gameOver = true;
				// 방금 둔 사람이 AI 이므로
				if (state.getWinningStatus() == WinningStatus::LOSE) {
					resultText = "Minimax WINS!";
				}
				else if (state.getWinningStatus() == WinningStatus::WIN) {
					resultText = "Montecarlo WIN!";
				}
				else {
					resultText = "DRAW";
				}
			}
			else {
				humanTurn = false;  // 다시 사람 차례
			}
		}

		// ----------- 렌더링 -----------
		BeginDrawing();
		ClearBackground(RAYWHITE);

		// 격자 그리기
		for (int y = 0; y < H; y++)
		{
			for (int x = 0; x < W; x++)
			{
				int px = PAD + x * CELL;
				int py = PAD + y * CELL;
				DrawRectangleLines(px, py, CELL, CELL, BLACK);
			}
		}

		// 보드 상태 문자열로 가져오기
		std::string s = state.toString();
		// CR 제거 (윈도우 방어)
		s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());

		std::stringstream ss(s);
		std::string line;

		// 첫 줄: is_first:\t0/1 버리기
		std::getline(ss, line);

		// 이후 H줄: 실제 보드
		std::vector<std::string> rows;
		rows.reserve(H);
		for (int i = 0; i < H; ++i) {
			if (std::getline(ss, line)) {
				rows.push_back(line);
			}
		}

		// rows[0] = 최상단 줄, rows[H-1] = 최하단 줄
		for (int y = 0; y < H; y++)
		{
			for (int x = 0; x < W; x++)
			{
				if (y >= (int)rows.size() || x >= (int)rows[y].size()) continue;

				char c = rows[y][x];  // '.', 'x', 'o'

				int cx = PAD + x * CELL + CELL / 2;
				int cy = PAD + y * CELL + CELL / 2;

				if (c == 'x')       DrawCircle(cx, cy, CELL * 0.35f, BLACK);
				else if (c == 'o')  DrawCircle(cx, cy, CELL * 0.35f, RED);
			}
		}

		// 상태 텍스트
		if (!gameOver) {
			const char* turnText = humanTurn ? "Your turn" : "AI thinking...";
			DrawText(turnText, 10, 10, 20, DARKBLUE);
		}
		else {
			DrawText(resultText.c_str(), 10, 10, 30, MAROON);
		}

		EndDrawing();
	}

	CloseWindow();
}
