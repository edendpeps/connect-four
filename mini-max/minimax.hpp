#pragma once
#include "ConnectFourState.hpp"


extern double duration;
int negamaxAction(const ConnectFourState& state, int depth, int time_limit_ms);
