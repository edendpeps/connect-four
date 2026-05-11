#pragma once
#include "ConnectFourState.hpp"


extern double duration;
int negamaxAction(const ConnectFourState& state, int depth, int time_limit_ms);
void open_data_file(const std::string& filename);
void open_data_files(const std::string& train_filename, const std::string& valid_filename);
void close_data_file();
void save_sample(const ConnectFourState& s, bool to_validation = false);
int alphaBetaAction(const ConnectFourState& state, int max_depth, int time_limit_ms);
