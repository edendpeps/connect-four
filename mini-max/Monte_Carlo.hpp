#pragma once
#include "ConnectFourState.hpp"
#include <random>

extern std::mt19937 mt_for_action;
extern double duration;

int randomAction(const ConnectFourState& state);
int MontecarloAction(const ConnectFourState& state, int playout_num, int time_limit_ms);
