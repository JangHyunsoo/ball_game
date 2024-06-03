#pragma once
#include <SDL.h>

enum class KeyPress
{
    NONE,
    PRESS,
    HOLD,
    RELEASE
};

class InputManager
{
public:
    static InputManager& getInstance() {
        static InputManager instance;
        return instance;
    }
private:
    InputManager() {}
    InputManager(InputManager const& other) = delete;
    void operator=(InputManager const& other) = delete;

private:
    KeyPress leftMouse = KeyPress::NONE;
    KeyPress rightMouse = KeyPress::NONE;
    int mouse_x;
    int mouse_y;
    SDL_Keycode keyCode;
public:
    bool isLeftMouse(KeyPress keyState) {
        return leftMouse == keyState;
    }
    bool isRightMouse(KeyPress keyState) {
        return rightMouse == keyState;
    }
    void getMousePos(int* x, int* y) {
        *x = mouse_x;
        *y = mouse_y;
    }
    int getX() {
        return mouse_x;
    }
    int getY() {
        return mouse_y;
    }
    bool isPressKey(SDL_Keycode _key) {
        return _key == keyCode;
    }

public:
    bool init() {
        leftMouse = KeyPress::NONE;
        rightMouse = KeyPress::NONE;
        mouse_x = 0;
        mouse_y = 0;
        return true;
    }
    bool update(const SDL_Event& event) {
        switch (event.type) {
        case SDL_MOUSEBUTTONDOWN:
            if (event.button.button == SDL_BUTTON_LEFT) {
                leftMouse = KeyPress::PRESS;
            }
            if (event.button.button == SDL_BUTTON_RIGHT) {
                rightMouse = KeyPress::PRESS;
            }
            SDL_GetMouseState(&mouse_x, &mouse_y);
            break;
        case SDL_MOUSEBUTTONUP:
            if (event.button.button == SDL_BUTTON_LEFT) {
                leftMouse = KeyPress::RELEASE;
            }
            if (event.button.button == SDL_BUTTON_RIGHT) {
                rightMouse = KeyPress::RELEASE;
            }
            SDL_GetMouseState(&mouse_x, &mouse_y);
            break;
        case SDL_MOUSEMOTION:
            if (leftMouse == KeyPress::HOLD || rightMouse == KeyPress::HOLD) {
                SDL_GetMouseState(&mouse_x, &mouse_y);
            }
            break;
        case SDL_KEYDOWN:
            keyCode = event.key.keysym.sym;
        }

        return true;
    }

    void next() {
        if (leftMouse == KeyPress::PRESS) leftMouse = KeyPress::HOLD;
        else if (leftMouse == KeyPress::RELEASE) leftMouse = KeyPress::NONE;
        if (rightMouse == KeyPress::PRESS) rightMouse = KeyPress::HOLD;
        else if (rightMouse == KeyPress::RELEASE) rightMouse = KeyPress::NONE;
        if (keyCode != SDLK_UNKNOWN) keyCode = SDLK_UNKNOWN;
    }
};

