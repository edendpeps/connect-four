//미니맥스 완성 후 머지 및 푸쉬 완료, 몬테카를로 만들차례
#include "minimax.hpp"
#include "ConnectFourState.hpp"
#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <iostream>
#include <chrono>
#include<random>
using State = ConnectFourState;
std::random_device rnd;
std::mt19937 mt_for_action(rnd());
static constexpr double INF = 100000000;

int randomAction(const State& state)
{
	auto legal_actions = state.legalActions();
	return legal_actions[mt_for_action() % (legal_actions.size())];
}
double playout(State state)
{
	// 게임 끝났으면, "현재 시점의 플레이어" 기준으로 평가
	if (state.isDone())
	{
		switch (state.getWinningStatus())
		{
		case WinningStatus::LOSE:
			// 현재 플레이어 입장: LOSE = 내가 이긴 상태
			return 0.0;
		case WinningStatus::WIN:
			// 현재 플레이어 입장: WIN  = 내가 진 상태
			return 1.0;
		case WinningStatus::DRAW:
			return 0.5;
		default:
			return 0.5;
		}
	}

	// 아직 안 끝났으면, 랜덤으로 한 수 두고
	state.advance(randomAction(state));
	// 턴이 바뀌었으니까, 값도 뒤집어 주기
	return 1.0 - playout(state);
}
int MontecarloAction(const State& state, int playout_num, int time_limit_ms)
{
	TimeKeeper tk(time_limit_ms);

	auto start = std::chrono::high_resolution_clock::now();
	auto legal_actions = state.legalActions();
	auto values = std::vector<double>(legal_actions.size());
	auto cnts = std::vector<double>(legal_actions.size());
	int playnum = 0;
	for (int cnt = 0; cnt < playout_num; cnt++)
	{
		if (tk.isTimeOver()) break;
		int index = cnt % legal_actions.size();
		State next_state = state;
		next_state.advance(legal_actions[index]);
		values[index] += 1.0 - playout(next_state);
		++cnts[index];
		playnum = cnts[index];

	}
	int best_action_index = -1;
	double best_score = -INF;
	for (int index = 0; index < legal_actions.size(); index++)
	{
		if (cnts[index] == 0.0)
		{
			continue;
		}

		double value_mean = values[index] / cnts[index];
		if (value_mean > best_score)
		{
			best_score = value_mean;
			best_action_index = index;
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration<double, std::milli>(end - start).count();
	std::cout << "----------------------------------montecarlo----------------------------------\n";
	std::cout << "playout_num: " << playnum << "\n";
	std::cout << "duration: " << duration << "\n";
	return legal_actions[best_action_index];
}
