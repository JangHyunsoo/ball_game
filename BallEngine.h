#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include "InputManager.h"
#include "Ball.h"
#include "cuda_partition.h"

using namespace std;

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
	int screen_width_;
	int screen_height_;

private:
	vector<Ball> vecBalls;
	int selected_idx_ = -1;

	void AddBall(float x, float y, float r)
	{
		Ball b;
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
	bool init(int _width, int _height, int _circle_count)
	{
		float fDefaultRad = 8.0f;
		selected_idx_ = -1;
		screen_width_ = _width;
		screen_height_ = _height;

		AddBall(_width / 2, _height / 2, 20);

		for (int i = 0; i < _circle_count; i++)
			AddBall(rand() % screen_width_, rand() % screen_height_, rand() % 4 + 2);

		initBallCuda(vecBalls);

		return true;
	}

	bool update(double deltaTime) {

		if (InputManager::getInstance().isLeftMouse(KeyPress::PRESS) || InputManager::getInstance().isRightMouse(KeyPress::PRESS))
		{
			selected_idx_ = -1;
			int mouse_x = InputManager::getInstance().getX();
			int mouse_y = InputManager::getInstance().getY();

			int selected_idx = selectBallCuda(mouse_x, mouse_y);
			if (selected_idx != -1) {
				selected_idx_ = selected_idx;
			}
		}

		if (InputManager::getInstance().isLeftMouse(KeyPress::HOLD))
		{
			if (selected_idx_ != -1)
			{
				int mouse_x = InputManager::getInstance().getX();
				int mouse_y = InputManager::getInstance().getY();
				Ball ball = vecBalls[selected_idx_];
				ball.px = mouse_x;
				ball.py = mouse_y;
				setBallArray(ball.id, ball);
			}
		}

		if (InputManager::getInstance().isLeftMouse(KeyPress::RELEASE))
		{
			selected_idx_ = -1;
		}

		if (InputManager::getInstance().isRightMouse(KeyPress::RELEASE))
		{
			if (selected_idx_ != -1)
			{
				int mouse_x = InputManager::getInstance().getX();
				int mouse_y = InputManager::getInstance().getY();
				Ball ball = vecBalls[selected_idx_];
				ball.vx = 5.0f * ((ball.px) - (float)mouse_x);
				ball.vy = 5.0f * ((ball.py) - (float)mouse_y);
				setBallArray(ball.id, ball);
			}

			selected_idx_ = -1;
		}

		moveBallCuda(screen_width_, screen_height_, deltaTime);
		vecBalls = collisionBallCuda();

		return true;
	}

	void render(SDL_Renderer* renderer) {
		SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);

		for (auto ball : vecBalls) {
			drawCircle(renderer, ball.px, ball.py, ball.radius);
		}

		SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);

		if (selected_idx_ != -1) {
			int mouse_x = InputManager::getInstance().getX();
			int mouse_y = InputManager::getInstance().getY();
			Ball ball = vecBalls[selected_idx_];
			SDL_RenderDrawLine(renderer, (int)ball.px, (int)ball.py, mouse_x, mouse_y);
		}
	}

	void destory() {
		freeBallCuda();
	}
};

