#include "raylib.h"
#include "ConnectFourState.hpp"
#include "minimax.hpp"
#include "Monte_Carlo.hpp"
#include <algorithm>
#include<iostream>
#include <string>
#include<sstream>

// GUI 설정
static constexpr int CELL = 40;
static constexpr int PAD = 20;
const int INF = 1000000000;
int main()
{
    const int screenW = W * CELL + PAD * 2;
    const int screenH = H * CELL + PAD * 2;

    InitWindow(screenW, screenH, "Gomoku (using your C++ AI)");
    SetTargetFPS(60);

    ConnectFourState state;

    bool humanTurn = true; // 사람 선공(필요하면 바꿔)
    int minimax_depth = 30;
    int time_limit = 3000;   // ms
    int playout_num = INF;

    while (!WindowShouldClose())
    {
        // --- 입력(사람) ---
        if (!state.isDone() && humanTurn && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
        {
            int mx = GetMouseX() - PAD;
            int my = GetMouseY() - PAD;
            if (0 <= mx && 0 <= my)
            {
                int x = mx / CELL;
                int y = my / CELL;
                if (0 <= x && x < W && 0 <= y && y < H)
                {
                    int action = y * W + x;
                    auto legal = state.legalActions();
                    if (std::find(legal.begin(), legal.end(), action) != legal.end())
                    {
                        state.advance(action);
                        humanTurn = false;
                    }
                }
            }
            std::cout << "------------------------------------- 내 턴 -------------------------------------\n";
            std::cout << state.toString();
        }

        // --- AI 턴 ---
        if (!state.isDone() && !humanTurn && !IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
        {

            // 여기서 AI 골라서 쓰면 됨.
            //1) negamax
            int a = negamaxAction(state, minimax_depth, time_limit);
            std::cout << "AI action = " << a
                << " (y=" << a / W << ", x=" << a % W << ")\n";
            state.advance(a);
            //2) MonteCarlo 쓰고 싶으면 위 줄 대신 이거:
           //int a = MontecarloAction(state, playout_num, time_limit);
            
            humanTurn = true;
            std::cout << "------------------------------------- AI 턴 -------------------------------------\n";
            std::cout << duration;
            std::cout << state.toString();
        }

        // --- 렌더링 ---
        BeginDrawing();
        ClearBackground(RAYWHITE);

        // 격자
        for (int y = 0; y < H; y++)
        {
            for (int x = 0; x < W; x++)
            {
                int px = PAD + x * CELL;
                int py = PAD + y * CELL;
                DrawRectangleLines(px, py, CELL, CELL, BLACK);
            }
        }

        // --- 렌더링 ---
        std::string s = state.toString();

        // 1) CR 제거
        s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());

        // 2) 줄로 분리
        std::stringstream ss(s);
        std::string line;

        // 첫 줄(is_first) 버림
        std::getline(ss, line);

        bool is_first = true;
        {
            auto pos = line.find('\t');
            int v = 1;
            if (pos != std::string::npos) v = std::stoi(line.substr(pos + 1));
            is_first = (v != 0);
        }
        // 보드 줄 읽기
        std::vector<std::string> rows;
        rows.reserve(H);
        for (int i = 0; i < H; ++i) {
            std::getline(ss, line);
            rows.push_back(line);
        }
   
        // 3) rows[y][x] 기반으로 그리기
        for (int y = 0; y < H; y++)
        {
            for (int x = 0; x < W; x++)
            {
                char c = rows[y][x];

                int cx = PAD + x * CELL + CELL / 2;
                int cy = PAD + y * CELL + CELL / 2;
                if (c == 'x') DrawCircle(cx, cy, CELL * 0.35f, BLACK);
                else if (c == 'o') DrawCircle(cx, cy, CELL * 0.35f, RED);
            }
        }
        
        // 결과 텍스트
        if (state.isDone())
        {
            const char* msg = "DRAW";
            if (state.getWinningStatus() == WinningStatus::LOSE) msg = "Previous player WIN!";
            DrawText(msg, 10, 10, 30, BLUE);
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
