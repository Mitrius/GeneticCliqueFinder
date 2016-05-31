#include <iostream>
#include <ctime>
#include "../Headers/CliqueFinder.h"
#include <fstream>
int main() {
    srand((unsigned int) time(NULL));
    std::string filename;
    std::cin>>filename;
	std::vector<Entry> history;
    Graph graph(filename);
	bool good = true;
	int i = 0;
	for (auto featDesc :graph.featDescriptorArray){
		for (unsigned int k = 0; k < featDesc.length();k++){
			if ((featDesc[k] > 122 && featDesc[k] < 65) && (featDesc[k] != ':') && (featDesc[k] != ' ')){
				good = false;
				break;
			}				
		}
		if (!good)
			continue;
		CliqueFinder finder(graph, 20, 10, i, 50);
		i++;
		history.push_back(finder.start());
		good = true;
	}
	filename = "results.txt";
	std::ofstream outSTream(filename, std::ofstream::trunc);
	for (auto entry : history){
		entry.print(outSTream);
	}

    return 0;
}
