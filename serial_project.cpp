#include <SDL.h>
#include <iostream>
#include <string>
#include <vector>
#include "SDLEngine.h"
#include "cuda_partition.h"

using namespace std;


int main(int argc, char* argv[]) {
    if (!SDLEngine::getInstance().initTest(1600, 900)) return false;
    else {
        SDLEngine::getInstance().logic();
    }
    SDLEngine::getInstance().destory();
    return 0;
}

