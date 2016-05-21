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
    void crossOver(std::vector<Organism> &pop);
    int getWorth(Organism pop);
    int RyBKA(std::set<int> &r, std::set<int> &p, std::set<int> &x);
    std::vector<int> randPerm(unsigned int size);
    void selection(std::vector<Organism> currentPop,std::vector<Organism> newPop);
    void nextGeneration();
    std::vector<Organism> population;
    const Graph* graph;
    int epoch = 0;
    int cliqueFeat = 0;
    double pMut = 0.4;

    std::vector<Organism> start();
    CliqueFinder(const Graph &g, int startAmount, unsigned int startSize, int feat);
};


#endif //NSAP_CLIQUEFINDER_H
