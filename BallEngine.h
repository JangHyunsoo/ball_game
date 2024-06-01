#pragma once
#include <vector>
#include <cmath>
#include "InputManager.h"

using namespace std;

struct sBall
{
    float px, py; // 중앙
    float vx, vy; // 속도
    float ax, ay; // 가속
    float radius; // 반지름
    float mass; // 무게

    int id; // idx
};

class BallEngine
{
public:
	static BallEngine& getInstance() {
		static BallEngine instance;
		return instance;
	}
private:
	BallEngine() {}
	BallEngine(BallEngine const& other) = delete;
	void operator=(BallEngine const& other) = delete;


private:
	int screenWidth = 640;
	int screenHeight = 640;

private:
	vector<sBall> vecBalls;
	sBall* pSelectedBall = nullptr;

	void AddBall(float x, float y, float r = 5.0f)
	{
		sBall b;
		b.px = x; b.py = y;
		b.vx = 0; b.vy = 0;
		b.ax = 0; b.ay = 0;
		b.radius = r;
		b.mass = r * 10.0f;

		b.id = vecBalls.size();
		vecBalls.emplace_back(b);
	}
	bool doCirclesOverlap(float x1, float y1, float r1, float x2, float y2, float r2)
	{
		return fabs((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) <= (r1 + r2) * (r1 + r2);
	}	
	bool isPointInCircle(float x1, float y1, float r1, float px, float py)
	{
		return fabs((x1 - px) * (x1 - px) + (y1 - py) * (y1 - py)) < (r1 * r1);
	}
	void drawCircle(SDL_Renderer* renderer, int cx, int cy, int radius) {
		for (int w = 0; w < radius * 2; w++) {
			for (int h = 0; h < radius * 2; h++) {
				int dx = radius - w;
				int dy = radius - h;
				if ((dx * dx + dy * dy) <= (radius * radius)) {
					SDL_RenderDrawPoint(renderer, cx + dx, cy + dy);
				}
			}
		}
	}

public:
	bool init(int circle_count = 100)
	{
		float fDefaultRad = 8.0f;

		for (int i = 0; i < circle_count; i++)
			AddBall(rand() % screenWidth, rand() % screenHeight, rand() % 16 + 2);

		return true;
	}

	bool update(float deltaTime) {

		if (InputManager::getInstance().isLeftMouse(KeyPress::PRESS) || InputManager::getInstance().isRightMouse(KeyPress::PRESS))
		{
			pSelectedBall = nullptr;
			int mouse_x = InputManager::getInstance().getX();
			int mouse_y = InputManager::getInstance().getY();

			for (auto& ball : vecBalls)
			{
				if (isPointInCircle(ball.px, ball.py, ball.radius, mouse_x, mouse_y))
				{
					pSelectedBall = &ball;
					break;
				}
			}
		}

		if (InputManager::getInstance().isLeftMouse(KeyPress::HOLD))
		{
			if (pSelectedBall != nullptr)
			{
				int mouse_x = InputManager::getInstance().getX();
				int mouse_y = InputManager::getInstance().getY();
				pSelectedBall->px = mouse_x;
				pSelectedBall->py = mouse_y;
			}
		}

		if (InputManager::getInstance().isLeftMouse(KeyPress::RELEASE))
		{
			pSelectedBall = nullptr;
		}

		if (InputManager::getInstance().isRightMouse(KeyPress::RELEASE))
		{
			if (pSelectedBall != nullptr)
			{
				int mouse_x = InputManager::getInstance().getX();
				int mouse_y = InputManager::getInstance().getY();
				pSelectedBall->vx = 5.0f * ((pSelectedBall->px) - (float)mouse_x);
				pSelectedBall->vy = 5.0f * ((pSelectedBall->py) - (float)mouse_y);
			}

			pSelectedBall = nullptr;
		}
		
		vector<pair<sBall*, sBall*>> vecCollidingPairs;

		for (auto& ball : vecBalls)
		{
			ball.ax = -ball.vx * 0.8f;
			ball.ay = -ball.vy * 0.8f;

			ball.vx += ball.ax * deltaTime;
			ball.vy += ball.ay * deltaTime;
			ball.px += ball.vx * deltaTime;
			ball.py += ball.vy * deltaTime;

			if (ball.px < 0) ball.px += (float)screenWidth;
			if (ball.px >= screenWidth) ball.px -= (float)screenWidth;
			if (ball.py < 0) ball.py += (float)screenHeight;
			if (ball.py >= screenHeight) ball.py -= (float)screenHeight;

			if (fabs(ball.vx * ball.vx + ball.vy * ball.vy) < 0.01f)
			{
				ball.vx = 0;
				ball.vy = 0;
			}
		}


		for (auto& ball : vecBalls)
		{
			for (auto& target : vecBalls)
			{
				if (ball.id != target.id)
				{
					if (doCirclesOverlap(ball.px, ball.py, ball.radius, target.px, target.py, target.radius))
					{
						vecCollidingPairs.push_back({ &ball, &target });

						float fDistance = sqrtf((ball.px - target.px) * (ball.px - target.px) + (ball.py - target.py) * (ball.py - target.py));

						float fOverlap = 0.5f * (fDistance - ball.radius - target.radius);

						ball.px -= fOverlap * (ball.px - target.px) / fDistance;
						ball.py -= fOverlap * (ball.py - target.py) / fDistance;

						target.px += fOverlap * (ball.px - target.px) / fDistance;
						target.py += fOverlap * (ball.py - target.py) / fDistance;
					}
				}
			}
		}

		for (auto c : vecCollidingPairs)
		{
			sBall* b1 = c.first;
			sBall* b2 = c.second;

			float fDistance = sqrtf((b1->px - b2->px) * (b1->px - b2->px) + (b1->py - b2->py) * (b1->py - b2->py));

			float nx = (b2->px - b1->px) / fDistance;
			float ny = (b2->py - b1->py) / fDistance;

			float tx = -ny;
			float ty = nx;

			float dpTan1 = b1->vx * tx + b1->vy * ty;
			float dpTan2 = b2->vx * tx + b2->vy * ty;

			float dpNorm1 = b1->vx * nx + b1->vy * ny;
			float dpNorm2 = b2->vx * nx + b2->vy * ny;

			float m1 = (dpNorm1 * (b1->mass - b2->mass) + 2.0f * b2->mass * dpNorm2) / (b1->mass + b2->mass);
			float m2 = (dpNorm2 * (b2->mass - b1->mass) + 2.0f * b1->mass * dpNorm1) / (b1->mass + b2->mass);

			b1->vx = tx * dpTan1 + nx * m1;
			b1->vy = ty * dpTan1 + ny * m1;
			b2->vx = tx * dpTan2 + nx * m2;
			b2->vy = ty * dpTan2 + ny * m2;
		}

		return true;
	}

	void render(SDL_Renderer* renderer) {
		SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);

		for (auto ball : vecBalls) {
			drawCircle(renderer, ball.px, ball.py, ball.radius);
		}

		SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);

		if (pSelectedBall != nullptr) {
			int mouse_x = InputManager::getInstance().getX();
			int mouse_y = InputManager::getInstance().getY();
			SDL_RenderDrawLine(renderer, (int)pSelectedBall->px, (int)pSelectedBall->py, mouse_x, mouse_y);
		}
	}


};

