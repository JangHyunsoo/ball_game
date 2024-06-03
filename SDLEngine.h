#pragma once
#include <SDL.h>
#include <string>
#include <iostream>
#include "InputManager.h"
#include "BallEngine.h"

class SDLEngine
{
public:
	static SDLEngine& getInstance() {
		static SDLEngine instance;
		return instance;
	}
private:
	SDLEngine() {}
	SDLEngine(SDLEngine const& other) = delete;
	void operator=(SDLEngine const& other) = delete;
private:
	SDL_Window* window_;
	SDL_Renderer* renderer_;
	bool running_ = true;

	int width_;
	int height_;
	char* window_name_;
public:
	bool init(char* _window_name, int _width, int _height) {
		if (!initSDL(_window_name, _width, _height)) return false;
		if (!initManager()) return false;

		return true;
	}
	bool logic() {
		SDL_Event event;
		Uint64 lastTime = SDL_GetPerformanceCounter();

		while (running_) {
			while (SDL_PollEvent(&event)) {
				switch (event.type) {
				case SDL_QUIT:
					running_ = false;
					break;
				}
				InputManager::getInstance().update(event);
			}

			Uint64 currentTime = SDL_GetPerformanceCounter();
			double deltaTime = (double)((currentTime - lastTime) / (double)SDL_GetPerformanceFrequency());
			lastTime = currentTime;

			SDL_SetWindowTitle(window_, ("FPS: " + std::to_string(1 / deltaTime)).c_str());

			update(deltaTime);

			InputManager::getInstance().next();

			sdl_render();
		}

		return true;
	}

	bool update(float deltaTime) {
		BallEngine::getInstance().update(deltaTime);
		return true;
	}

	void sdl_render() {
		SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
		SDL_RenderClear(renderer_);

		render(renderer_);

		SDL_RenderPresent(renderer_);
	}
	
	void render(SDL_Renderer* _renderer) {
		BallEngine::getInstance().render(_renderer);
	}

	void destory() {
		SDL_DestroyRenderer(renderer_);
		SDL_DestroyWindow(window_);
		SDL_Quit();
	}
private:
	bool initSDL(char* _window_name, int _width, int _height) {
		if (SDL_Init(SDL_INIT_VIDEO) != 0) {
			return false;
		}

		width_ = _width;
		height_ = _height;
		window_name_ = _window_name;

		window_ = SDL_CreateWindow(_window_name, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, _width, _height, SDL_WINDOW_SHOWN);
		renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_ACCELERATED);

		return true;
	}

	bool initManager() {
		if (!InputManager::getInstance().init()) {
			cout << "InputManager init Error...\n";
			return false;
		}
		if (!BallEngine::getInstance().init(width_, height_, 5000)) {
			cout << "BallEngine init Error...\n";
			return false;
		}
		return true;
	}
};

