#include <SDL.h>
#include <iostream>
#include <string>
#include <vector>
#include "SDLEngine.h"

using namespace std;


int main(int argc, char* argv[]) {
    if (!SDLEngine::getInstance().init("Ball Game", 1600, 900)) return false;
    else {
        SDLEngine::getInstance().logic();
    }
    SDLEngine::getInstance().destory();
    return 0;
}
