#include <iostream>
#include <ctime>
#include "../Headers/CliqueFinder.h"
#include "cuda_runtime.h"

int main() {
    srand((unsigned int) time(NULL));
    std::string filename;
    std::cin>>filename;
	std::vector<std::pair<Organism,int> > history;
    Graph graph(filename);
	bool good = true;
	for (auto featDesc :graph.featDescriptorArray){
		for (int k = 0; k < featDesc.length();k++){
			if ((featDesc[k] > 122 && featDesc[k] < 65) && (featDesc[k] != ':') && (featDesc[k] != ' ')){
				good = false;
				break;
			}				
		}
		if (!good)
			continue;
		CliqueFinder finder(graph, 20, 100, 1, 100);
		auto res = finder.start();
		history.push_back(res);
		std::cout << "Dla cechy: " << graph.featDescriptorArray[1] << " Rozmiar kliki wynosi " 
			<< res.first.worth << "Rozmiar niepe³nej kliki wynosi: " << res.second << std::endl;
		good = true;
	}
    
    

    return 0;
}
