#define __CUDACC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include <iostream>
#include "Ball.h"

#define BLOCK_SIZE 8

float* px;
float* py;
float* vx;
float* vy;
float* ax;
float* ay;
float* radius;
float* mass;
Ball* device_balls;
int n;

std::vector<Ball> deviceArrayToVector(Ball* device_array, int n) {
    std::vector<Ball> host_vector(n);

    cudaMemcpy(host_vector.data(), device_array, n * sizeof(Ball), cudaMemcpyDeviceToHost);

    return host_vector;
}

void initBallCuda(const std::vector<Ball>& _host_balls) {
    n = _host_balls.size();
    cudaMalloc(&device_balls, n * sizeof(Ball));
    cudaMemcpy(device_balls, _host_balls.data(), n * sizeof(Ball), cudaMemcpyHostToDevice);
}

void freeBallCuda() {
    cudaFree(device_balls);
}

__device__ bool isPointInCircle(float x1, float y1, float r1, float px, float py)
{
    return fabs((x1 - px) * (x1 - px) + (y1 - py) * (y1 - py)) < (r1 * r1);
}

__global__ void checkSelectBall(Ball* balls, int n, int mouse_x, int mouse_y, int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    Ball ball = balls[idx];

    if (threadIdx.x == 0) {
        *result = -1;
    }

    __syncthreads();

    if (isPointInCircle(ball.px, ball.py, ball.radius, mouse_x, mouse_y))
    {
        // atomicMin(result, idx);
        *result = idx;
    }
}

int selectBallCuda(int mouse_x, int mouse_y) {
    int host_selected_idx = -1;
    int* device_selected_idx;
    cudaMalloc(&device_selected_idx, sizeof(int));
    cudaMemcpy(&device_selected_idx, &host_selected_idx, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    checkSelectBall <<<blocksPerGrid, threadsPerBlock>>> (device_balls, n, mouse_x, mouse_y, device_selected_idx);
    cudaMemcpy(&host_selected_idx, device_selected_idx, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_selected_idx);

    return host_selected_idx;
}

__global__ void moveBall(Ball* balls, int n, int width, int height, float deltaTime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    balls[idx].ax = -balls[idx].vx * 0.8f;
    balls[idx].ay = -balls[idx].vy * 0.8f;

    balls[idx].vx += balls[idx].ax * deltaTime;
    balls[idx].vy += balls[idx].ay * deltaTime;
    balls[idx].px += balls[idx].vx * deltaTime;
    balls[idx].py += balls[idx].vy * deltaTime;

    if (balls[idx].px < 0) balls[idx].px += (float)width;
    if (balls[idx].px >= width) balls[idx].px -= (float)width;
    if (balls[idx].py < 0) balls[idx].py += (float)height;
    if (balls[idx].py >= height) balls[idx].py -= (float)height;

    if (fabs(balls[idx].vx * balls[idx].vx + balls[idx].vy * balls[idx].vy) < 0.01f)
    {
        balls[idx].vx = 0;
        balls[idx].vy = 0;
    }
}

void moveBallCuda(int width, int height, double deltaTime) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    moveBall <<<blocksPerGrid, threadsPerBlock >>> (device_balls, n, width, height, (float)deltaTime);
}

__global__ void setBall(Ball* balls, int n, int idx, Ball ball) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        balls[idx] = ball;
    }
}

void setBallArray(int idx, Ball ball) {
    setBall << <1, 1 >> > (device_balls, n, idx, ball);
}

__device__ bool doCirclesOverlap(float x1, float y1, float r1, float x2, float y2, float r2)
{
    return fabs((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) <= (r1 + r2) * (r1 + r2);
}

__global__ void collisionBall(Ball* balls, int n) {
    int one = blockIdx.x * blockDim.x + threadIdx.x;
    int other = blockIdx.y * blockDim.y + threadIdx.y;

    if (other < n && one < n && other > one) {
        Ball& ball = balls[one];
        Ball& target = balls[other];


        if (doCirclesOverlap(ball.px, ball.py, ball.radius, target.px, target.py, target.radius))
        {
            // static
            float fDistance = sqrtf((ball.px - target.px) * (ball.px - target.px) + (ball.py - target.py) * (ball.py - target.py));

            float fOverlap = 0.5f * (fDistance - ball.radius - target.radius);

            ball.px -= fOverlap * (ball.px - target.px) / fDistance;
            ball.py -= fOverlap * (ball.py - target.py) / fDistance;

            target.px += fOverlap * (ball.px - target.px) / fDistance;
            target.py += fOverlap * (ball.py - target.py) / fDistance;

            // dynamic
            float fChnagedDistance = sqrtf((ball.px - target.px) * (ball.px - target.px) + (ball.py - target.py) * (ball.py - target.py));

            float nx = (target.px - ball.px) / fChnagedDistance;
            float ny = (target.py - ball.py) / fChnagedDistance;

            float tx = -ny;
            float ty = nx;

            float dpTan1 = ball.vx * tx + ball.vy * ty;
            float dpTan2 = target.vx * tx + target.vy * ty;

            float dpNorm1 = ball.vx * nx + ball.vy * ny;
            float dpNorm2 = target.vx * nx + target.vy * ny;

            float m1 = (dpNorm1 * (ball.mass - target.mass) + 2.0f * target.mass * dpNorm2) / (ball.mass + target.mass);
            float m2 = (dpNorm2 * (target.mass - ball.mass) + 2.0f * ball.mass * dpNorm1) / (ball.mass + target.mass);

            ball.vx = tx * dpTan1 + nx * m1;
            ball.vy = ty * dpTan1 + ny * m1;
            target.vx = tx * dpTan2 + nx * m2;
            target.vy = ty * dpTan2 + ny * m2;
        }

        __syncthreads();
    }
}

std::vector<Ball> collisionBallCuda() {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    int grid_size = ((n % BLOCK_SIZE) > 0 ? 1 : 0) + n / BLOCK_SIZE;
    dim3 grid(grid_size, grid_size);

    collisionBall <<<grid, block >>> (device_balls, n);

    return deviceArrayToVector(device_balls, n);
}