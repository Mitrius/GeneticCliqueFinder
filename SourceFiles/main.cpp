#include <iostream>
#include <ctime>
#include "../Headers/CliqueFinder.h"
#include "cuda_runtime.h"

int main() {
	cudaSetDevice(1);
    srand((unsigned int) time(NULL));
    std::string filename;
    std::cin>>filename;
    Graph graph(filename);
    CliqueFinder finder(graph, 20, 100, 1, 100);
    auto res = finder.start();
    std::cout << res.first.worth << std::endl;
    return 0;
}
