#pragma once
#include "ConnectFourState.hpp"

int MCTSAction(const ConnectFourState& state, int playout_number, int time_limit_ms);
void open_data_files(const std::string& train_filename, const std::string& validation_filename);
void set_data_split(bool is_validation);
void close_data_files();
void save_sample(const ConnectFourState& s);
void open_policy_file(const std::string& policy_filename);
void close_policy_file();