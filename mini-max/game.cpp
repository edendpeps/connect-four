#include<vector>
#include <random>
//ĳ���� ����
struct Character {
	int y_;
	int x_;
	int game_score_;

	Character(const int y = 0, const int x = 0) : y_(y), x_(x), game_score_(0) {}
};

enum class WinningStatus {
	WIN,
	LOSE,
	DRAW,
	NONE
};

//������
// �� �� �� �� ���� (�Ǵ� �װ� ���ϴ� ���� ����)
constexpr int dx[4] = { 0, 1, 0, -1 };
constexpr int dy[4] = { -1, 0, 1, 0 };
constexpr const int H = 3;
constexpr const int W = 3;
constexpr const int END_TURN = 4;

class Game
{
private:
	std::vector<std::vector<int>> points_;
	int turn_;
	std::vector<Character> characters_;
public:
	// ���� ����
	Game(const int seed) : points_(H, std::vector<int>(W)), turn_(0), characters_({ Character(H / 2,(W / 2) - 1), Character(H / 2, (W / 2) + 1) })
	{
		auto mt_for_construct = std::mt19937(seed);
		for (int y = 0; y < H; y++)
		{
			for (int x = 0; x < W; x++)
			{
				int point = mt_for_construct() % 10;
				if (characters_[0].y_ == y && characters_[0].x_ == x)
				{
					continue;
				}
				if (characters_[1].y_ == y && characters_[1].x_ == x)
				{
					continue;
				}
				this->points_[y][x] = point;
			}
		}
	}
	//���� ���� ����
	bool isDone() const
	{
		return this->turn_ == END_TURN;
	}
	//������ �ൿ���� �����ϰ� ���� �÷��̾� ��ȯ
	void advance(const int action)
	{
		auto& character = this->characters_[0];
		character.x_ += dx[action];
		character.y_ += dy[action];
		auto& point = this->points_[character.y_][character.x_];
		if (point > 0)
		{
			character.game_score_ += point;
			point = 0;
		}
		this->turn_++;
		std::swap(this->characters_[0], this->characters_[1]);
	}
	//�÷��̾ ������ �ൿ ��� ȹ��
	std::vector<int> legalActions() const
	{
		std::vector<int> actions;
		const auto& character = this->characters_[0];
		for (int action = 0; action < 4; action++)
		{
			int ty = character.y_ + dy[action];
			int tx = character.x_ + dx[action];
			if (ty >= 0 && ty < H && tx >= 0 && tx < W)
			{
				actions.emplace_back(action);
			}
		}
		return actions;
	}

	//����
	WinningStatus getWinningStatus() const
	{
		if (isDone())
		{
			if (characters_[0].game_score_ > characters_[1].game_score_) 
			{
				return WinningStatus::WIN;
			}
			else if (characters_[0].game_score_ < characters_[1].game_score_) 
			{
				return WinningStatus::LOSE;
			}
			else
			{
				return WinningStatus::DRAW;
			}
		}
		else
		{
			return WinningStatus::NONE;
		}
	}

};
