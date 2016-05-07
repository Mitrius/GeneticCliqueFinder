#include <iostream>
#include "../Headers/CliqueFinder.h"

int main() {
    std::string filename;
    std::cin>>filename;
    Graph graph(filename);
    CliqueFinder finder(graph,10,2);
    return 0;
}
