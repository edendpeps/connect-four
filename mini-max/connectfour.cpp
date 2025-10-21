#include<vector>
//ĳ���� ����
struct Character {
	int y_;
	int x_;
	int gamescore;
	Character(const int y = 0, const int x = 0) : y_(y), x_(x), gamescore(0) {}
};


//������
constexpr const int H = 3;
constexpr const int W = 3;

enum class WinningStatus {
	WIN,
	LOSE,
	DRAW,
	NONE
};

class ConnectFourState {
private:
	bool first_ = true;
	int my_board_[H][W] = {};
	int enemy_board_[H][W] = {};
	WinningStatus winning_status_ = WinningStatus::NONE;
public:
	// ���� ����
	ConnectFourState()
	{
	}
	
	//���� ���� ����
	bool isDone() const
	{
		return winning_status_ != WinningStatus::NONE;
	}
	//�������� ȹ��
	WinningStatus getWinningStatus() const
	{
		return this->winning_status_;
	}
	//�÷��̾ ������ �ൿ ��� ȹ��
	std::vector<int> legalActions() const
	{
		std::vector<int> actions;
		for (int x = 0; x < W; x++)
		{
			for (int y = H - 1; y >= 0; y--)
			{
				if (my_board_[y][x] == 0 && enemy_board_[y][x] == 0)
				{
					actions.emplace_back(x);
					break;
				}
			}
		}
		return actions;
	}
	void advance(const int action)
	{
		std::pair<int, int> coordinate;
		for (int y = 0; y < H; y++)
		{
			if (this->my_board_[y][action] == 0 && this->enemy_board_[y][action] == 0)
			{
				this->my_board_[y][action] = 1;
				coordinate = std::pair<int, int>(y, action);
				break;
			}
		}
	}
};
	