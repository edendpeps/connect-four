#pragma once
#include "ConnectFourState.hpp"


extern double duration;
int negamaxAction(const ConnectFourState& state, int depth, int time_limit_ms);
void open_data_file(const std::string& filename);
void close_data_file();
void save_sample(const ConnectFourState& s);
int alphaBetaAction(const ConnectFourState& state, int max_depth, int time_limit_ms);