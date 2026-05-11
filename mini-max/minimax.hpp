#pragma once
#include "ConnectFourState.hpp"


extern double duration;
int negamaxAction(const ConnectFourState& state, int depth, int time_limit_ms);
void open_data_files(const std::string& train_filename, const std::string& validation_filename);
void set_data_split(bool is_validation);
void close_data_files();
void save_sample(const ConnectFourState& s);
int alphaBetaAction(const ConnectFourState& state, int max_depth, int time_limit_ms);
