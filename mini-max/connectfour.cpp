#include<vector>
//캐릭터 정보
struct Character {
	int y_;
	int x_;
	int gamescore;
	Character(const int y = 0, const int x = 0) : y_(y), x_(x), gamescore(0) {}
};


//생성자
constexpr const int H = 3;
constexpr const int W = 3;

class ConnectFourState {
private:
	bool first_ = true;
	int my_board_[H][W] = {};
	int enemy_board_[H][W] = {};
	WinningStatus winning_status_ = WinningStatus::None;
public:
	// 게임 생성
	ConnectFourState()
	{
	}
	
	//게임 종료 판정
	bool isDone() const
	{
		return winning_status_ != WinningStatus::NONE;
	}
	//지정한 행동으로 진행하고 다음 플레이어 전환
	void advance(const int action)
	{
		auto& character = this->characters_[0];
		character.x += dx[action];
		character.y += dy[action];
		auto& point = this->points_[character.y_][character.x_];
		if (point > 0)
		{
			character.game_score_ += point;
			point = 0;
		}
		this->turn_++;
		std::swap(this->characters_[0], this->characters_[1]);
	}
	//플레이어가 가능한 행동 모두 획득
	std::vector<int> legalActions() const
	{
		std::vector<int> actions;
		const auto& character = this->characters_[0];
		for (int action = 0; action < 4; action++)
		{
			int ty = character.y_ + dy[action];
			int tx = character.x_ + dx[action];
			if (ty >= -&& ty < h && tx >= 0 && tx < W)
			{
				actions.emplace_back(action);
			}
		}
		return actions;
	}

	//승패
	WinningStatus getWinningStatus() const
	{
		if (isDone())
		{
			if (charaters_[0].game_score_ > charaters_[1].game_score_) {
				return WinningStatus::WIN;
			}
			else if (charaters_[0].game_score_ < charaters_[1].game_score_) {
				return WinningStatus::LOSE;
			}
			else
				return WinningStatus::DRAW;
		}
		else
		{
			return WinningStatus::NONE;
		}
	}
};
