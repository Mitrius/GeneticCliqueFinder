#include <string>
#include <algorithm>
#include <fstream>
#include "Organism.h"

struct Entry{
	Organism winner;
	std::string featName;
	int possibleClique;
	int cliqueNumber;
	void print(std::ofstream &stream){
		std::string output = featName + "  ";
		output += std::to_string(winner.worth) + "  ";
		for (auto vert : winner.vertices){
			output += std::to_string(vert) + " ";
		}
		output += " " + std::to_string(possibleClique);
		output += " " + std::to_string(cliqueNumber) + '\n';
		stream << output;
	}
};