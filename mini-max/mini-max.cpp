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

ConnectFourState::ConnectFourState() {} // 커넥트포 클래스 참조
using State = ConnectFourState; // 커넥트포 

using ScoreType = int64_t;
constexpr const ScoreType INF = 1000000000LL;

int evaluate(const State& s) {
	// 1) 종결 우선
	if (s.isDone()) {
		switch (s.getWinningStatus()) {
		case WinningStatus::LOSE: return 100000000;  // 현재 시점(두는 쪽) 승리
		case WinningStatus::WIN:  return -100000000; // 현재 시점 패배
		default: return 0; // DRAW
		}
	}

	static const int WIN_SCORE = 100000000;
	static const int THREE_OPEN = 1000;
	static const int THREE_OPEN_BLOCK = 1200;
	static const int TWO_OPEN = 50;
	static const int CENTER_BONUS_PER_PIECE = 6;

	int score = 0;

	// 중앙 보너스
	const int centerX = W / 2;
	for (int y = 0; y < H; ++y) {
		// 내 말은 'x'/'o' 표기가 아니라 내부적으로 my_board==1이므로
		// 문자열을 파싱할 필요 없이, 출력용 char 없이 평가하려면
		// State 내부 보드를 못 보니 toString으로 대체하기 어렵다.
		// → 간단히: 보드 문자로 읽기

	}

	// ---- 권장: 보드 접근이 필요하니 State에 보드 읽기용 getter를 추가하는 게 정석.
	// 지금 구조를 그대로 간다면 toString()으로 문자를 읽어오는 트릭을 쓸 수도 있으나 비효율.
	// 아래는 toString()을 이용한 안전한(하지만 느린) 버전 예시:

	auto boardStr = s.toString();
	// boardStr은 맨 위줄부터 찍힘. 파싱 편의상 2차원 배열로 재구성
	// 형식: 첫줄 "is_first:\tX\n", 이후 H줄의 보드, 각 줄 W문자 + '\n'
	std::vector<std::string> rows;
	{
		std::stringstream ss(boardStr);
		std::string line;
		std::getline(ss, line); // is_first 라인 버림
		for (int i = 0; i < H; ++i) {
			std::getline(ss, line);
			rows.push_back(line);
		}
	}
	// rows[0]가 최상단 줄(시각화 상단), rows[H-1]이 바닥. 좌표 변환 주의.
	auto at = [&](int y, int x)->char { return rows[H - 1 - y][x]; }; // y=0 바닥이 되도록.

	auto score_window = [&](int y0, int x0, int dy, int dx) {
		int my = 0, opp = 0, emp = 0;
		for (int k = 0; k < 4; k++) {
			char c = at(y0 + k * dy, x0 + k * dx);
			if (c == 'x') my++;
			else if (c == 'o') opp++;
			else emp++;
		}
		if (my == 4) return WIN_SCORE;
		if (opp == 4) return -WIN_SCORE;
		int s = 0;
		if (my == 3 && emp == 1) s += THREE_OPEN;
		if (my == 2 && emp == 2) s += TWO_OPEN;
		if (opp == 3 && emp == 1) s -= THREE_OPEN_BLOCK;
		return s;
		};

	// 2) 윈도우 스캔 (가로)
	for (int y = 0; y < H; y++)
		for (int x = 0; x <= W - 4; x++)
			score += score_window(y, x, 0, 1);
	// 세로
	for (int y = 0; y <= H - 4; y++)
		for (int x = 0; x < W; x++)
			score += score_window(y, x, 1, 0);
	// 대각 
	for (int y = 0; y <= H - 4; y++)
		for (int x = 0; x <= W - 4; x++)
			score += score_window(y, x, 1, 1);
	// 대각 /
	for (int y = 3; y < H; y++)
		for (int x = 0; x <= W - 4; x++)
			score += score_window(y, x, -1, 1);

	// 3) 중앙 가산 (문자 보드 기준)
	for (int y = 0; y < H; y++) {
		if (at(y, centerX) == 'x') score += CENTER_BONUS_PER_PIECE;
	}

	return score;
}

int minimax(State state, int depth, bool maximizingPlayer) {
	if (depth == 0 || state.isDone()) return evaluate(state);

	auto actions = state.legalActions();
	if (maximizingPlayer) {
		int best = -INF;
		for (int a : actions) {
			State child = state;
			child.advance(a);
			int val = minimax(child, depth - 1, false);
			if (val > best) best = val;
		}
		return best;
	}
	else {
		int best = INF;
		for (int a : actions) {
			State child = state;
			child.advance(a);
			int val = minimax(child, depth - 1, true);
			if (val < best) best = val;
		}
		return best;
	}
}

int minimaxAction(const State& state, int depth) {
	auto actions = state.legalActions();
	int bestA = actions.front();
	int bestV = -INF;
	for (int a : actions) {
		State child = state;
		child.advance(a);
		int v = minimax(child, depth - 1, false);
		if (v > bestV) { bestV = v; bestA = a; }
	}
	return bestA;
}
