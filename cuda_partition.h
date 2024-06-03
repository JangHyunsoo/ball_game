#pragma once
void initBallCuda(const std::vector<Ball>& _host_balls);
void freeBallCuda();
int selectBallCuda(int mouse_x, int mouse_y);
void moveBallCuda(int width, int height, double deltaTime);
void setBallArray(int idx, Ball ball);
void testCuda(int idx, float value);
std::vector<Ball> collisionBallCuda();