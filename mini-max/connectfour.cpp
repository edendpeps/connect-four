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
#include "MCTS.h"
#include <set>
#include <limits> // 추가: 입력 유효성 처리에 필요
#include <fstream>

int monte_win = 0;
int minimax_win = 0;
int minimax2_win = 0;
int game_draw = 0;
int puremc_win = 0;
int game_limit = 2000;
// 시간을 관리하는 클래스 
const int time_limit = 500;
const int time_limit_minimax = 500;
const int INF = 100000000;
const int minimax_depth = INF;
const int roll_out = INF;

ConnectFourState::ConnectFourState() {}

bool ConnectFourState::isDone() const {
	return winning_status_ != WinningStatus::NONE;
}
enum class OpponentType {
	AlphaBeta,
	MCTS,
	PMC
};

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

#include <fstream>
#include <array>

static std::ofstream value_train_file;
static std::ofstream value_val_file;
static std::ofstream value_test_file;
static std::ofstream* current_value_file = nullptr;

struct ValueSample {
	std::array<int, 42> board;
	bool to_move_is_first;
};

std::array<int, 42> encode_state_for_value(const State& s) {
	std::array<int, 42> encoded{};

	const int(*my)[W] = s.getMyBoard();
	const int(*opp)[W] = s.getEnemyBoard();

	int idx = 0;
	for (int y = 0; y < H; y++) {
		for (int x = 0; x < W; x++) {
			if (my[y][x] == 1) encoded[idx++] = 1;
			else if (opp[y][x] == 1) encoded[idx++] = -1;
			else encoded[idx++] = 0;
		}
	}

	return encoded;
}

void write_value_header(std::ofstream& fout) {
	for (int i = 0; i < 42; i++) {
		fout << "c" << i << ",";
	}
	fout << "value\n";
}

void open_value_files(
	const std::string& train_filename,
	const std::string& val_filename,
	const std::string& test_filename
) {
	value_train_file.open(train_filename);
	value_val_file.open(val_filename);
	value_test_file.open(test_filename);

	write_value_header(value_train_file);
	write_value_header(value_val_file);
	write_value_header(value_test_file);

	current_value_file = &value_train_file;
}
enum class DataSplit {
	Train,
	Val,
	Test
};

void set_value_split(DataSplit split) {
	if (split == DataSplit::Train) {
		current_value_file = &value_train_file;
	}
	else if (split == DataSplit::Val) {
		current_value_file = &value_val_file;
	}
	else {
		current_value_file = &value_test_file;
	}
}

void close_value_files() {
	if (value_train_file.is_open()) value_train_file.close();
	if (value_val_file.is_open()) value_val_file.close();
}

void save_value_row(const std::array<int, 42>& board, int value) {
	if (current_value_file == nullptr || !current_value_file->is_open()) return;

	for (int i = 0; i < 42; i++) {
		(*current_value_file) << board[i] << ",";
	}
	(*current_value_file) << value << "\n";
}
bool last_move_by_minimax = false; // 직전에 누가 뒀는지
void playGame(bool first_is_minimax, OpponentType opponent)
{
	int turn_count = 0;

	auto state = State();
	//std::cout << state.toString() << "\n";

	std::vector<ValueSample> pending_samples;

	while (!state.isDone())
	{
		if (turn_count >= 4) {
			pending_samples.push_back({
				encode_state_for_value(state),
				state.isFirst()
				});
		}
		bool minimax_turn = (turn_count % 2 == 0) == first_is_minimax;

		if (minimax_turn)
		{
			//std::cout << "alphabeta1 ------------------------------------\n";
			int action = alphaBetaAction(state, minimax_depth, time_limit_minimax);
			//std::cout << "Turn : " << turn_count << "\n";
			//std::cout << "action " << action << "\n";
			state.advance(action);
		}
		else
		{
			int action;
			if (opponent == OpponentType::AlphaBeta)
			{

				//std::cout << "alphabeta2 ---------------------------------\n";
				action = alphaBetaAction(state, minimax_depth, time_limit_minimax);
				//std::cout << "Turn : " << turn_count << "\n";
				//std::cout << "action " << action << "\n";
			}
			else if (opponent == OpponentType::MCTS)
			{
				//std::cout << "MCTS ---------------------------------\n";
				action = MCTSAction(state, roll_out, time_limit_minimax);
				//std::cout << "Turn : " << turn_count << "\n";
				//std::cout << "action " << action << "\n";
			}
			else
			{
				//std::cout << "PureMC ---------------------------------\n";
				action = MontecarloAction(state, roll_out, time_limit_minimax);
				//std::cout << "Turn : " << turn_count << "\n";
				//std::cout << "action " << action << "\n";
			}
			state.advance(action);
		}

		last_move_by_minimax = minimax_turn;
		//std::cout << state.toString() << "\n";
		turn_count++;
	}

	if (state.getWinningStatus() == WinningStatus::DRAW)
	{
		//std::cout << "DRAW\n";
		game_draw++;
	}
	else if (state.getWinningStatus() == WinningStatus::LOSE)
	{
		// 직전에 둔 사람이 이김
		//std::cout << "winner: " << (last_move_by_minimax ? "alphabeta" : "ab2") << "\n";
		if (last_move_by_minimax) minimax_win++;
		else
		{
			if (opponent == OpponentType::MCTS)
			{
				monte_win++;
			}
			else if (opponent == OpponentType::PMC)
			{
				puremc_win++;
			}
			else
			{
				minimax2_win++;
			}
		}
	}
	else if (state.getWinningStatus() == WinningStatus::WIN)
	{
		// 직전에 둔 사람이 짐 -> 상대가 이김
		//std::cout << "winner: " << (last_move_by_minimax ? "ab2" : "alphabeta") << "\n";
		if (last_move_by_minimax)
		{
			if (opponent == OpponentType::MCTS)
			{
				monte_win++;
			}
			else if (opponent == OpponentType::PMC)
			{
				puremc_win++;
			}
			else
			{
				minimax2_win++;
			}
		}
		else
		{
			minimax_win++;
		}
	}

	bool draw = (state.getWinningStatus() == WinningStatus::DRAW);
	bool winner_is_first = false;

	if (!draw) {
		if (state.getWinningStatus() == WinningStatus::LOSE) {
			// advance() 후에는 다음 플레이어 관점으로 바뀌므로
			// LOSE는 방금 둔 이전 플레이어가 이겼다는 뜻
			winner_is_first = !state.isFirst();
		}
		else if (state.getWinningStatus() == WinningStatus::WIN) {
			// 현재 구조에서는 거의 안 나오지만 안전용
			winner_is_first = state.isFirst();
		}
	}

	for (const auto& sample : pending_samples) {
		int value = 0;

		if (!draw) {
			value = (sample.to_move_is_first == winner_is_first) ? 1 : -1;
		}

		save_value_row(sample.board, value);
	}
}

int main()
{
	open_value_files(
		"C:/Users/User/Desktop/value_train.csv",
		"C:/Users/User/Desktop/value_val.csv",
		"C:/Users/User/Desktop/value_test.csv"
	);

	const int training_game_limit = static_cast<int>(game_limit * 0.7);
	const int validation_game_limit = static_cast<int>(game_limit * 0.15);
	const int test_game_limit = game_limit - training_game_limit - validation_game_limit;
	const int training_half = training_game_limit / 2;
	const int validation_half = validation_game_limit / 2;

	for (int i = 0; i < game_limit; i++) {
		DataSplit split;
		int split_index;

		if (i < training_game_limit) {
			split = DataSplit::Train;
			split_index = i;
		}
		else if (i < training_game_limit + validation_game_limit) {
			split = DataSplit::Val;
			split_index = i - training_game_limit;
		}
		else {
			split = DataSplit::Test;
			split_index = i - training_game_limit - validation_game_limit;
		}

		set_value_split(split);

		std::cout << "\n---------------------- game_count: " << i << " [";

		if (split == DataSplit::Train) std::cout << "training";
		else if (split == DataSplit::Val) std::cout << "validation";
		else std::cout << "test";

		std::cout << "]\n";

		OpponentType opponent;

		int r = split_index % 100;

		if (r < 65) opponent = OpponentType::AlphaBeta;
		else if (r < 90) opponent = OpponentType::MCTS;
		else opponent = OpponentType::PMC;

		bool first_is_minimax = (i % 2 == 0);
		playGame(first_is_minimax, opponent);
		if (i == 1999)
		{
			std::cout << "alphabeta_win: " << minimax_win << "\n";
			std::cout << "MCTS_win: " << monte_win << "\n";
			std::cout << "alphabeta2_win: " << minimax2_win << "\n";
			std::cout << "pureMC_win: " << puremc_win << "\n";

			std::cout << "Draw: " << game_draw;;

		}
	}
}