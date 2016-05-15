#ifndef NSAP_CLIQUEFINDER_H
#define NSAP_CLIQUEFINDER_H

#include <vector>
#include "Organism.h"
#include "Graph.h"
/*
 * Class representing logic of algorithm
 * All organisms from population represent the same feat
 */
class CliqueFinder {
public:
    void crossOver(std::vector<Organism> &pop, unsigned long childrenAmount);

    int getWorth(Organism pop);
    std::vector<int> randPerm(unsigned int size);

    void selection(std::vector<Organism> &newPop);
    void nextGeneration();
    std::vector<Organism> population;
    const Graph* graph;
    int epoch = 0;
    int maxEpoch;
    int cliqueFeat = 0;
    double pMut = 0.4;

    std::pair<Organism, int> start();

    CliqueFinder(const Graph &g, const int startAmount, const unsigned int startSize, const int feat,
                 const int desMaxEpoch);
};


#endif //NSAP_CLIQUEFINDER_H
