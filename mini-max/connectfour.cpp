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
<<<<<<< HEAD
#include <limits> // Ãß°¡: ÀÔ·Â À¯È¿¼º Ã³¸®¿¡ ÇÊ¿ä
#include <fstream>
=======
#include <limits> // ì¶”ê?: ?…ë ¥ ? íš¨??ì²˜ë¦¬???„ìš”
>>>>>>> b556b8ae7804023c794f8cabd1e502b630c57d0b

int monte_win = 0;
int minimax_win = 0;
int minimax2_win = 0;
int game_draw = 0;
<<<<<<< HEAD
int puremc_win = 0;
int game_limit = 2000;
// ½Ã°£À» °ü¸®ÇÏ´Â Å¬·¡½º 
const int time_limit = 500;
const int time_limit_minimax = 500;
=======
int game_limit = 1000;
// ?œê°„??ê´€ë¦¬í•˜???´ë˜??
const int time_limit = 100 + rand() % 900;;
const int time_limit_minimax = 100 + rand() % 900;;
>>>>>>> b556b8ae7804023c794f8cabd1e502b630c57d0b
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
<<<<<<< HEAD

// helper: (y,x)¿¡¼­ (dy,dx) ¹æÇâÀ¸·Î ³» µ¹ ¿¬¼Ó ±æÀÌ
=======
// helper: (y,x)?ì„œ (dy,dx) ë°©í–¥?¼ë¡œ ?????°ì† ê¸¸ì´
>>>>>>> b556b8ae7804023c794f8cabd1e502b630c57d0b
inline int run(const int board[H][W], int y, int x, int dy, int dx) {
	int cnt = 0;
	while (y >= 0 && y < H && x >= 0 && x < W && board[y][x] == 1) {
		++cnt; y += dy; x += dx;
	}
	return cnt;
}

void ConnectFourState::advance(const int action)
{
	// 1. ë§??“ê¸°
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
		auto acts = legalActions();    // ?„ì¬ ??ê¸°ì??¼ë¡œ ????ê³??ˆëŠ”ì§€
		board_full = acts.empty();
	}

	// 2. ???˜ê¸°ê¸?(??ƒ)
	std::swap(my_board_, enemy_board_);
	is_first_ = !is_first_;

	// 3. ?´ì œ state??"?¤ìŒ?????¬ëŒ" ê´€?ì´??
	if (win_now) {
		// ë°©ê¸ˆ ???¬ëŒ???´ê²¼?¼ë‹ˆê¹? ì§€ê¸?state ?…ì¥?ì„  ?´ê? ì§?ê²?
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

<<<<<<< HEAD
#include <fstream>
#include <array>
=======
// ë¬´ì‘???‰ë™
>>>>>>> b556b8ae7804023c794f8cabd1e502b630c57d0b

static std::ofstream value_train_file;
static std::ofstream value_val_file;
static std::ofstream value_test_file;
static std::ofstream* current_value_file = nullptr;

<<<<<<< HEAD
struct ValueSample {
	std::array<int, 42> board;
	bool to_move_is_first;
};
=======
// ?¬ëŒ ?…ë ¥(1P): ??ë²ˆí˜¸ë¥??…ë ¥ë°›ì•„ ê²€ì¦?
int humanAction(const State& state)
{
	using std::cout;
	using std::cin;
	using std::endl;
>>>>>>> b556b8ae7804023c794f8cabd1e502b630c57d0b

std::array<int, 42> encode_state_for_value(const State& s) {
	std::array<int, 42> encoded{};

<<<<<<< HEAD
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
bool last_move_by_minimax = false; // Á÷Àü¿¡ ´©°¡ µ×´ÂÁö
=======
	int col;
	while (true)
	{
		//cout << "?¹ì‹ ??ì°¨ë??…ë‹ˆ??(??ë²ˆí˜¸ 0~" << (W - 1) << "): ";
		if (!(cin >> col))
		{
			cin.clear();
			cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			//cout << "?«ìë¥??…ë ¥?˜ì„¸??\n";
			continue;
		}
		if (0 <= col && col < W && ok.count(col))
		{
			return col;
		}
		cout << "ê·??´ì? ?????†ìŠµ?ˆë‹¤. ê°€?¥í•œ ?? ";
		for (auto c : legal) cout << c << " ";
		cout << endl;
	}
}

// ê²Œì„??1???Œë ˆ?? 1P(?¬ëŒ), 2P(?œë¤ AI)
	bool last_move_by_minimax = false; // ì§ì „???„ê? ?€?”ì?
>>>>>>> b556b8ae7804023c794f8cabd1e502b630c57d0b
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
<<<<<<< HEAD
		// Á÷Àü¿¡ µĞ »ç¶÷ÀÌ ÀÌ±è
		//std::cout << "winner: " << (last_move_by_minimax ? "alphabeta" : "ab2") << "\n";
=======
		// ì§ì „?????¬ëŒ???´ê?
		std::cout << "winner: " << (last_move_by_minimax ? "alphabeta" : "ab2") << "\n";
>>>>>>> b556b8ae7804023c794f8cabd1e502b630c57d0b
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
<<<<<<< HEAD
		// Á÷Àü¿¡ µĞ »ç¶÷ÀÌ Áü -> »ó´ë°¡ ÀÌ±è
		//std::cout << "winner: " << (last_move_by_minimax ? "ab2" : "alphabeta") << "\n";
		if (last_move_by_minimax)
=======
	const int training_game_limit = static_cast<int>(game_limit * 0.8);
	const int validation_game_limit = game_limit - training_game_limit;
	const int training_half = training_game_limit / 2;
	const int validation_half = validation_game_limit / 2;

		const bool is_validation = (i >= training_game_limit);
		const int split_index = is_validation ? (i - training_game_limit) : i;

		std::cout << "\n\n---------------------- game_count: " << i
			<< " [" << (is_validation ? "validation" : "training") << "]\n\n";

		if (!is_validation) {
			// training 80% interval: MCTS : AlphaBeta = 1 : 1
			opponent = (split_index < training_half) ? OpponentType::AlphaBeta : OpponentType::MCTS;
		}
		else {
			// validation 20% interval: MCTS : AlphaBeta = 1 : 1
			opponent = (split_index < validation_half) ? OpponentType::AlphaBeta : OpponentType::MCTS;
		}

>>>>>>> b556b8ae7804023c794f8cabd1e502b630c57d0b
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
			// advance() ÈÄ¿¡´Â ´ÙÀ½ ÇÃ·¹ÀÌ¾î °üÁ¡À¸·Î ¹Ù²î¹Ç·Î
			// LOSE´Â ¹æ±İ µĞ ÀÌÀü ÇÃ·¹ÀÌ¾î°¡ ÀÌ°å´Ù´Â ¶æ
			winner_is_first = !state.isFirst();
		}
		else if (state.getWinningStatus() == WinningStatus::WIN) {
			// ÇöÀç ±¸Á¶¿¡¼­´Â °ÅÀÇ ¾È ³ª¿ÀÁö¸¸ ¾ÈÀü¿ë
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
<<<<<<< HEAD

		int r = split_index % 100;

		if (r < 65) opponent = OpponentType::AlphaBeta;
		else if (r < 90) opponent = OpponentType::MCTS;
		else opponent = OpponentType::PMC;

		bool first_is_minimax = (i % 2 == 0);
=======
		if (i % 10 < 7) opponent = OpponentType::AlphaBeta; // 70%
		else opponent = OpponentType::MCTS;                 // 30%
		bool first_is_minimax = (i % 2 == 0); // ë²ˆê°ˆ??? ê³µ
>>>>>>> b556b8ae7804023c794f8cabd1e502b630c57d0b
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