#pragma once
#include "ConnectFourState.hpp"

extern double duration;
void order_actions(std::vector<int>& acts);   // ← 이거 추가!

int negamaxAction(const ConnectFourState& state, int depth, int time_limit_ms);
