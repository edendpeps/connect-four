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

std::random_device rnd;
std::mt19937 mt_for_action(0);

int randomAction(const State& state)
{
	auto legal_actions = state.legalActions();
	return legal_actions[mt_for_action() % (legal_actions.size())];
}
using State = ConnectFourState;
double playout(State* state)
{
	switch (state->getWinningStatus())
	{
	case(WinningStatus::WIN):
		return 1.;
	case(WinningStatus::LOSE):
		return 0.;
	case(WinningStatus::DRAW):
		return 0.5;
	defualt:
		state->advance(randomAction(*state));
		return 1. - playout(state);
	}
	
}
int MontecarloAction(const State& state, int playout_num)
{
	auto legal_actions = state.legalActions();
	auto values = std::vector<double>(legal_actions.size());
	auto cnts = std::vector<double>(legal_actions.size());
	for (int cnt = 0; cnt < playout_num; cnt++)
	{
		int index = cnt % legal_actions.size();
	}

}
