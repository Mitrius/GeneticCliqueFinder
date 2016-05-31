#include <iostream>
#include <ctime>
#include "../Headers/CliqueFinder.h"
#include <fstream>

struct Entry{
	std::pair<Organism, int> result;
	std::string featName;
	void print(std::ofstream &stream){
		std::string output = featName+ "  ";
		output += std::to_string(result.first.worth)+ "  ";
		for (auto vert : result.first.vertices){
			output += std::to_string(vert) + " ";
		}
		output += " " + std::to_string(result.second);
		stream << output;
	}
};
int main() {
	cudaSetDevice(1);
    srand((unsigned int) time(NULL));
    std::string filename;
    std::cin>>filename;
	std::vector<Entry> history;
    Graph graph(filename);
	bool good = true;
	Entry ent;
	for (auto featDesc :graph.featDescriptorArray){
		for (unsigned int k = 0; k < featDesc.length();k++){
			if ((featDesc[k] > 122 && featDesc[k] < 65) && (featDesc[k] != ':') && (featDesc[k] != ' ')){
				good = false;
				break;
			}				
		}
		if (!good)
			continue;
		CliqueFinder finder(graph, 20, 100, 1, 100);
		ent.result = finder.start();
		ent.featName = featDesc;
		history.push_back(ent);
		good = true;
	}
	/*filename = "results.txt";
	std::ofstream outSTream(filename, std::ofstream::trunc);
	for (auto entry : history){
		entry.print(outSTream);
	}
    */

    return 0;
}
