#ifndef NSAP_CLIQUEFINDER_H
#define NSAP_CLIQUEFINDER_H

#include <vector>
#include "Organism.h"
#include "Graph.h"
/*
 * Class representing logic of algorithm
 *
 * All organisms from population represent the same feat
 */
class CliqueFinder {
public:
    Organism crossOver(Organism & a, Organism & b);
    int GetWorth(std::vector<Organism> pop);
    std::vector<int> randPerm(unsigned int size);
    void selection(std::vector<Organism> currentPop,std::vector<Organism> newPop);
    void nextGeneration();
    std::vector<Organism> population;
    Graph graph;
    int CliqueFeat = 0;
    double pMut = 0.4;


};


#endif //NSAP_CLIQUEFINDER_H