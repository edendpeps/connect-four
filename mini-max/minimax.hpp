#pragma once
#include "ConnectFourState.hpp"

extern double duration;
int negamaxAction(const ConnectFourState& state, int depth);
