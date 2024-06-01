#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include <iostream>
#include "Ball.h"

__device__ bool isPointInCircle(float x1, float y1, float r1, float px, float py)
{
    return fabs((x1 - px) * (x1 - px) + (y1 - py) * (y1 - py)) < (r1 * r1);
}

__global__ void checkSelectBall(Ball* balls, int n, int mouse_x, int mouse_y, int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Ball ball = balls[idx];

    if (idx >= n) return;

    if (isPointInCircle(ball.px, ball.py, ball.radius, mouse_x, mouse_y))
    {
        // atomicMin(result, idx);
        *result = idx;
    }
}

int selectBallCuda(const std::vector<Ball>& _host_balls, int mouse_x, int mouse_y) {
    int n = _host_balls.size();
    int host_selected_idx = 0;
    int* device_selected_idx;
    Ball* device_balls;
    cudaMalloc(&device_balls, n * sizeof(Ball));
    cudaMemcpy(device_balls, _host_balls.data(), n * sizeof(Ball), cudaMemcpyHostToDevice);

    cudaMalloc(&device_selected_idx, sizeof(int));
    cudaMemcpy(&device_selected_idx, &host_selected_idx, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    checkSelectBall <<<blocksPerGrid, threadsPerBlock>>> (device_balls, n, mouse_x, mouse_y, device_selected_idx);

    cudaMemcpy(&host_selected_idx, device_selected_idx, sizeof(int), cudaMemcpyDeviceToHost);

    return host_selected_idx;
}