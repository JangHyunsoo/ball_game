#define __CUDACC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include <iostream>
#include "Ball.h"

#define BLOCK_SIZE 32

float* device_px;
float* device_py;
float* device_vx;
float* device_vy;
float* device_ax;
float* device_ay;
float* device_radius;
float* device_mass;
Ball* device_balls;
int n;

__global__ void arrayToStructure(Ball* balls, float* px_vec, float* py_vec, float* vx_vec, float* vy_vec, float* ax_vec, float* ay_vec, float* radius_vec, float* mass_vec, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    If (idx >= n) Return;

    balls[idx].px = px_vec[idx];
    balls[idx].py = py_vec[idx];
    balls[idx].vx = vx_vec[idx];
    balls[idx].vy = vy_vec[idx];
    balls[idx].ax = ax_vec[idx];
    balls[idx].ay = ay_vec[idx];
    balls[idx].radius = radius_vec[idx];
    balls[idx].mass = mass_vec[idx];
    balls[idx].id = idx;

    //printf("%d  : %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", idx, px_vec[idx], balls[idx].py, balls[idx].vx, balls[idx].vy, balls[idx].ax, balls[idx].ay, balls[idx].mass, balls[idx].radius);
}

std: vector<Ball> deviceArrayToVector() {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<Ball> host_vector(n);

    arrayToStructure << <blocksPerGrid, threadsPerBlock >> > (device_balls, device_px, device_py, device_vx, device_vy, device_ax, device_ay, device_radius, device_mass, n);

    cudaMemcpy(host_vector.data(), device_balls, n * sizeof(Ball), cudaMemcpyDeviceToHost);

    return host_vector;
}

void initBallCuda(const std::vector<Ball>& _host_balls) {
    n = _host_balls.size();
    std::vector<float> px_vec(n);
    std::vector<float> py_vec(n);
    std::vector<float> vx_vec(n);
    std::vector<float> vy_vec(n);
    std::vector<float> ax_vec(n);
    std::vector<float> ay_vec(n);
    std::vector<float> radius_vec(n);
    std::vector<float> mass_vec(n);

    for (int i = 0; i <n; i++) {
        px_vec[i] = _host_balls[i].px;
        py_vec[i] = _host_balls[i].py;
        vx_vec[i] = _host_balls[i].vx;
        vy_vec[i] = _host_balls[i].vy;
        ax_vec[i] = _host_balls[i].ax;
        ay_vec[i] = _host_balls[i].ay;
        radius_vec[i] = _host_balls[i].radius;
        mass_vec[i] = _host_balls[i].mass;
    }

    cudaMalloc(& device_px, n * sizeof(float));
    cudaMemcpy(device_px, px_vec.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(& device_py, n * sizeof(float));
    cudaMemcpy(device_py, py_vec.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(& device_vx, n * sizeof(float));
    cudaMemcpy(device_vx, vx_vec.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(& device_vy, n * sizeof(float));
    cudaMemcpy(device_vy, vy_vec.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(& device_ax, n * sizeof(float));
    cudaMemcpy(device_ax, ax_vec.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(& device_ay, n * sizeof(float));
    cudaMemcpy(device_ay, ay_vec.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(& device_radius, n * sizeof(float));
    cudaMemcpy(device_radius, radius_vec.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(& device_mass, n * sizeof(float));
    cudaMemcpy(device_mass, mass_vec.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(& device_balls, n * sizeof(Ball));
    cudaMemcpy(device_balls, _host_balls.data(), n * sizeof(Ball), cudaMemcpyHostToDevice);
}

void freeBallCuda() {
    cudaFree(device_px);
    cudaFree(device_py);
    cudaFree(device_vx);
    cudaFree(device_vy);
    cudaFree(device_ax);
    cudaFree(device_ay);
    cudaFree(device_radius);
    cudaFree(device_mass);
    cudaFree(device_balls);
}

__device__ bool isPointInCircle(float x1, float y1, float r1, float px, float py)
{
    Return fabs((x1 - px) * (x1 - px) + (y1 - py) * (y1 - py)) < (r1 * r1);
}

__global__ void checkSelectBall(float* px_vec, float* py_vec, float* radius_vec, int n, int mouse_x, int mouse_y, int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    If (idx >= n) Return;

    int px = px_vec[idx];
    int py = py_vec[idx];
    int radius = radius_vec[idx];


    If (isPointInCircle(px, py, radius, mouse_x, mouse_y))
    {
        // atomicMin(result, idx);
        *result = idx;
    }
}

int selectBallCuda(int mouse_x, int mouse_y) {
    int host_selected_idx = 0;
    int* device_selected_idx;
    cudaMalloc(& device_selected_idx, sizeof(int));
    cudaMemset(device_selected_idx, -1, sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    checkSelectBall << <blocksPerGrid, threadsPerBlock >> > (device_px, device_py, device_radius, n, mouse_x, mouse_y, device_selected_idx);
    cudaMemcpy(&host_selected_idx, device_selected_idx, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_selected_idx);

    return host_selected_idx;
}

__global__ void moveBall(float* px_vec, float* py_vec, float* vx_vec, float* vy_vec, float* ax_vec, float* ay_vec, int n, int width, int height, float deltaTime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //printf("%.2f\n", px_vec[idx]);
    if (idx >= n) return;
    ax_vec[idx] = -vx_vec[idx] * 0.8f;
    ay_vec[idx] = -vy_vec[idx] * 0.8f;

    vx_vec[idx] += ax_vec[idx] * deltaTime;
    vy_vec[idx] += ay_vec[idx] * deltaTime;
    px_vec[idx] += vx_vec[idx] * deltaTime;
    py_vec[idx] += vy_vec[idx] * deltaTime;

    if (px_vec[idx] <0) px_vec[idx] += (float)width;
    if (px_vec[idx] >= width) px_vec[idx] -= (float)width;
    if (py_vec[idx] <0) py_vec[idx] += (float)height;
    if (py_vec[idx] >= height) py_vec[idx] -= (float)height;

    if (fabs(vx_vec[idx] * vx_vec[idx] + vy_vec[idx] * vy_vec[idx]) <0.01f)
    {
        vx_vec[idx] = 0;
        vy_vec[idx] = 0;
    }
}

void moveBallCuda(int width, int height, double deltaTime) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    moveBall << <blocksPerGrid, threadsPerBlock >> > (device_px, device_py, device_vx, device_vy, device_ax, device_ay, n, width, height, (float)deltaTime);
    
}

__global__ void setBall(float* px_vec, float* py_vec, float* vx_vec, float* vy_vec, float* ax_vec, float* ay_vec, float* radius_vec, float* mass_vec, int n, int idx, Ball ball) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        px_vec[idx] = ball.px;
        py_vec[idx] = ball.py;
        vx_vec[idx] = ball.vx;
        vy_vec[idx] = ball.vy;
        ax_vec[idx] = ball.ax;
        ay_vec[idx] = ball.ay;
        radius_vec[idx] = ball.radius;
        mass_vec[idx] = ball.mass;
    }
}

void setBallArray(int idx, Ball ball) {
    setBall << <1, 1 >> >(device_px, device_py, device_vx, device_vy, device_ax, device_ay, device_radius, device_mass, n, idx, ball);
}

__device__ bool doCirclesOverlap(float x1, float y1, float r1, float x2, float y2, float r2)
{
    return fabs((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) <= (r1 + r2) * (r1 + r2);
}

__global__ void collisionBall(float* px_vec, float* py_vec, float* vx_vec, float* vy_vec, float* radius_vec, float* mass_vec, int n) {
    int one = blockIdx.x * blockDim.x + threadIdx.x;
    int other = blockIdx.y * blockDim.y + threadIdx.y;

    if (other <n && one <n && other > one) {
        if (doCirclesOverlap(px_vec[one], py_vec[one], radius_vec[one], px_vec[other], py_vec[other], radius_vec[other]))
        {
            // static
            float fDistance = sqrtf((px_vec[one] - px_vec[other]) * (px_vec[one] - px_vec[other]) + (py_vec[one] - py_vec[other]) * (py_vec[one] - py_vec[other]));

            float fOverlap = 0.5f * (fDistance - radius_vec[one] - radius_vec[other]);

            px_vec[one] -= fOverlap * (px_vec[one] - px_vec[other]) / fDistance;
            py_vec[one] -= fOverlap * (py_vec[one] - py_vec[other]) / fDistance;

            px_vec[other] += fOverlap * (px_vec[one] - px_vec[other]) / fDistance;
            py_vec[other] += fOverlap * (py_vec[one] - py_vec[other]) / fDistance;

            // dynamic
            float fChnagedDistance = sqrtf((px_vec[one] - px_vec[other]) * (px_vec[one] - px_vec[other]) + (py_vec[one] - py_vec[other]) * (py_vec[one] - py_vec[other]));

            float nx = (px_vec[other] - px_vec[one]) / fChnagedDistance;
            float ny = (py_vec[other] - py_vec[one]) / fChnagedDistance;

            float tx = -ny;
            float ty = nx;

            float dpTan1 = vx_vec[one] * tx + vy_vec[one] * ty;
            float dpTan2 = vx_vec[other] * tx + vy_vec[other] * ty;

            float dpNorm1 = vx_vec[one] * nx + vy_vec[one] * ny;
            float dpNorm2 = vx_vec[other] * nx + vy_vec[other] * ny;

            float m1 = (dpNorm1 * (mass_vec[one] - mass_vec[other]) + 2.0f * mass_vec[other] * dpNorm2) / (mass_vec[one] + mass_vec[other]);
            float m2 = (dpNorm2 * (mass_vec[other] - mass_vec[one]) + 2.0f * mass_vec[one] * dpNorm1) / (mass_vec[one] + mass_vec[other]);

            vx_vec[one] = tx * dpTan1 + nx * m1;
            vy_vec[one] = ty * dpTan1 + ny * m1;
            vx_vec[other] = tx * dpTan2 + nx * m2;
            vy_vec[other] = ty * dpTan2 + ny * m2;
        }

        __syncthreads();
    }
}

std::vector<Ball> collisionBallCuda() {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    int grid_size = ((n % BLOCK_SIZE) > 0 ? 1 : 0) + n / BLOCK_SIZE;
    dim3 grid(grid_size, grid_size);

    collisionBall << <grid, block >> > (device_px, device_py, device_vx, device_vy, device_radius, device_mass, n);

    return deviceArrayToVector();
}

void testCuda(int idx, float value) {
    test << <1, 1 >> > (device_balls, n, idx, value);
}