#include <iostream>
#include <ctime>
#include "../Headers/CliqueFinder.h"

int main() {
    srand((unsigned int) time(NULL));
    std::string filename;
    std::cin>>filename;
    Graph graph(filename);
    //toast(graph);
    CliqueFinder finder(graph, 20, 100, 1, 100);
    auto res = finder.start();
    std::cout << res.first.worth << std::endl;
    return 0;
}
