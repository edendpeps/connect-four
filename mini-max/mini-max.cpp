#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <iostream>
#include <chrono>
#include "ConnectFourState.hpp"
#include "minimax.hpp"
#include <fstream>

using State = ConnectFourState; // 커넥트포 

// 평가 가중치(감각치, 필요 시 조정)
static constexpr int INF = 100000000;
static constexpr int WIN_SCORE = 100000000; // 종결 가중치(절대적으로 큼)
static constexpr int THREE_OPEN = 1000;      // 내 3 + 빈1
static constexpr int THREE_OPEN_BLOCK = 2000;      // 상대 3 + 빈1 (더 크게 패널티)
static constexpr int TWO_OPEN = 50;        // 내 2 + 빈2
static constexpr int CENTER_BONUS_PIECE = 6;         // 중앙 열 돌 1개당 보너스
double duration;

// toString()을 파싱해서 보드 접근: y=0이 바닥(게임 내부 좌표와 일치)
static inline void parseBoard(
	const State& s,
	std::vector<std::string>& rows,
	bool& is_first_out
) {
	rows.clear();
	std::stringstream ss(s.toString());
	std::string line;

	// 첫 줄: "is_first:\t0" 또는 "is_first:\t1"
	std::getline(ss, line);
	{
		auto pos = line.find('\t');
		int v = 0;
		if (pos != std::string::npos) {
			v = std::stoi(line.substr(pos + 1));
		}
		is_first_out = (v != 0);
	}

	// 이후 H개의 줄 = 보드
	for (int i = 0; i < H; ++i) {
		std::getline(ss, line);
		rows.push_back(line);
	}
}

static inline char at_cell(const std::vector<std::string>& rows, int y, int x) {
	// rows는 top→bottom이므로 뒤집어서 접근
	return rows[H - 1 - y][x]; // 'x' = 현재 플레이어, 'o' = 상대, '.' = 빈칸
}

// 4칸 윈도우 점수 계산
static inline int score_window(const std::vector<std::string>& rows,
	int y0, int x0, int dy, int dx) {
	int my = 0, opp = 0, emp = 0;
	for (int k = 0; k < 4; k++) {
		char c = at_cell(rows, y0 + k * dy, x0 + k * dx);
		if (c == 'x') my++;
		else if (c == 'o') opp++;
		else               emp++;
	}
	if (my == 4)  return  WIN_SCORE;
	if (opp == 4) return -WIN_SCORE;

	int s = 0;
	if (my == 3 && emp == 1) s += THREE_OPEN;
	if (my == 2 && emp == 2) s += TWO_OPEN;
	if (opp == 3 && emp == 1) s -= THREE_OPEN_BLOCK;
	return s;
}

// 현재 플레이어 관점 평가
static inline int eval(const State& s) {
	if (s.isDone()) {
		switch (s.getWinningStatus()) {
		case WinningStatus::LOSE:
			// 직전에 둔 쪽이 승리 → 지금 둘 차례인 쪽은 패배
			return -WIN_SCORE;
		case WinningStatus::WIN:
			// 직전에 둔 쪽이 패배 → 지금 둘 차례인 쪽이 승리
			return  WIN_SCORE;
		default:
			return 0;
		}
	}
	std::vector<std::string> rows;
	bool is_first = true;              // ← 선언
	parseBoard(s, rows, is_first);     // ← 여기서 채워짐

	// 후공 차례일 때(x/o 뒤집기)
	if (!is_first) {
		for (auto& row : rows) {
			for (auto& ch : row) {
				if (ch == 'x') ch = 'o';
				else if (ch == 'o') ch = 'x';
			}
		}
	}
	int score = 0;

	// 가로
	for (int y = 0; y < H; ++y)
		for (int x = 0; x <= W - 4; ++x)
			score += score_window(rows, y, x, 0, 1);

	// 세로
	for (int y = 0; y <= H - 4; ++y)
		for (int x = 0; x < W; ++x)
			score += score_window(rows, y, x, 1, 0);

	// 대각 
	for (int y = 0; y <= H - 4; ++y)
		for (int x = 0; x <= W - 4; ++x)
			score += score_window(rows, y, x, 1, 1);

	// 대각 /
	for (int y = 3; y < H; ++y)
		for (int x = 0; x <= W - 4; ++x)
			score += score_window(rows, y, x, -1, 1);

	// 중앙 열 가산
	const int cx = W / 2;
	for (int y = 0; y < H; ++y)
		if (at_cell(rows, y, cx) == 'x') score += CENTER_BONUS_PIECE;

	return score;
}

// 중앙 우선 정렬(선택)
static inline void order_actions(std::vector<int>& acts) {
	const int c = W / 2;
	std::sort(acts.begin(), acts.end(),
		[c](int a, int b) { return std::abs(a - c) < std::abs(b - c); });
}
static inline void order_actions_by_eval(const State& state, std::vector<int>& acts) {
	struct MoveScore {
		int action;
		int score;
	};

	std::vector<MoveScore> scored;
	scored.reserve(acts.size());

	for (int a : acts) {
		State child = state;
		child.advance(a);

		// child는 상대 차례 상태이므로 부모 입장 점수로 되돌리기 위해 -eval(child)
		int score = -eval(child);
		scored.push_back({ a, score });
	}

	std::sort(scored.begin(), scored.end(),
		[](const MoveScore& lhs, const MoveScore& rhs) {
			return lhs.score > rhs.score;
		});

	for (int i = 0; i < (int)acts.size(); ++i) {
		acts[i] = scored[i].action;
	}
}
// 네가맥스 (알파베타), 시간측정
int negamax_alpha_beta(State state, int depth, int alpha, int beta, TimeKeeper& tk) {
	if (depth == 0 || state.isDone()) return eval(state);

	if (tk.isTimeOver()) return eval(state);

	auto acts = state.legalActions();
	if (acts.empty()) return eval(state);

	order_actions_by_eval(state, acts);

	int best = -INF;

	for (int a : acts) {
		State child = state;
		child.advance(a);

		int v = -negamax_alpha_beta(child, depth - 1, -beta, -alpha, tk);

		if (v > best) best = v;
		if (v > alpha) alpha = v;

		// 가지치기
		if (alpha >= beta) break;
	}

	return best;
}
int alphaBetaAction(const State& state, int max_depth, int time_limit_ms)
{
	auto start = std::chrono::high_resolution_clock::now();
	TimeKeeper tk(time_limit_ms);

	int bestMove = -1;
	int depth = 1;

	while (depth <= max_depth && !tk.isTimeOver()) {
		auto acts = state.legalActions();
		if (acts.empty()) break;

		order_actions_by_eval(state, acts);

		int localBestMove = acts.front();
		int alpha = -INF;
		int beta = INF;

		for (int a : acts) {
			if (tk.isTimeOver()) break;

			State child = state;
			child.advance(a);

			int v = -negamax_alpha_beta(child, depth - 1, -beta, -alpha, tk);

			if (v > alpha) {
				alpha = v;
				localBestMove = a;
			}
		}

		if (!tk.isTimeOver()) {
			bestMove = localBestMove;
		}

		depth++;
	}

	if (bestMove == -1) {
		auto acts = state.legalActions();
		if (!acts.empty()) bestMove = acts.front();
	}

	auto end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration<double, std::milli>(end - start).count();
	std::cout << "alpha-beta depth: " << depth - 1 << "\n";
	return bestMove;
}

// 네가맥스 (알파베타 없음), 시간측정
int negamax(State state, int depth, TimeKeeper& tk) {
	if (depth == 0 || state.isDone()) return eval(state);

	if (tk.isTimeOver()) return eval(state);

	auto acts = state.legalActions();
	if (acts.empty()) return eval(state);

	order_actions(acts);

	int best = -INF;
	for (int a : acts) {
		State child = state;
		child.advance(a);

		int v = -negamax(child, depth - 1, tk);

		if (v > best) best = v;
	}
	return best;
}

// 최선 수 선택
int negamaxAction(const State& state, int max_depth, int time_limit_ms)
{
	auto start = std::chrono::high_resolution_clock::now();
	TimeKeeper tk(time_limit_ms);

	int bestMove = -1;
	int depth = 1;

	while (depth <= max_depth && !tk.isTimeOver()) {
		auto acts = state.legalActions();
		if (acts.empty()) break;

		order_actions(acts);

		int localBestMove = acts.front();
		int localBestV = -INF;

		for (int a : acts) {
			if (tk.isTimeOver()) break;

			State child = state;
			child.advance(a);

			int v = -negamax(child, depth - 1, tk);
			if (v > localBestV) {
				localBestV = v;
				localBestMove = a;
			}
		}

		// 이 깊이는 어느 정도 봤으니, 일단 이걸 “최신 최선 수”로 저장
		if (!tk.isTimeOver()) {
			bestMove = localBestMove;
		}

		depth++;
	}

	if (bestMove == -1) {
		// 시간 시작부터 거의 0ms였던 극단 케이스 방어
		auto acts = state.legalActions();
		if (!acts.empty()) bestMove = acts.front();
	}
	auto end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration<double, std::milli>(end - start).count();
	std::cout << "depth: " << depth - 1 << "\n";;
	return bestMove;
}
static std::ofstream data_file;

double normalize_score(double s) {
	double x = s / 10000.0;
	if (x > 1.0) x = 1.0;
	if (x < -1.0) x = -1.0;
	return x;
}

void open_data_file(const std::string& filename) {
	data_file.open(filename);
	for (int i = 0; i < 42; i++) {
		data_file << "c" << i << ",";
	}
	data_file << "score\n";
}

void close_data_file() {
	if (data_file.is_open()) data_file.close();
}

std::vector<int> encode_state(const State& s) {
	std::vector<int> encoded;
	encoded.reserve(42);

	std::vector<std::string> rows;
	bool is_first;
	parseBoard(s, rows, is_first);

	if (!is_first) {
		for (auto& row : rows) {
			for (auto& ch : row) {
				if (ch == 'x') ch = 'o';
				else if (ch == 'o') ch = 'x';
			}
		}
	}

	for (int y = 0; y < H; y++) {
		for (int x = 0; x < W; x++) {
			char c = at_cell(rows, y, x);
			if (c == 'x') encoded.push_back(1);
			else if (c == 'o') encoded.push_back(-1);
			else encoded.push_back(0);
		}
	}
	return encoded;
}

void save_sample(const State& s) {
	if (!data_file.is_open()) return;

	auto encoded = encode_state(s);
	double y = normalize_score(eval(s));

	for (int i = 0; i < 42; i++) {
		data_file << encoded[i] << ",";
	}
	data_file << y << "\n";
}