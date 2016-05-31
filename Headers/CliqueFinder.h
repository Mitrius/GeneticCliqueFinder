#ifndef NSAP_CLIQUEFINDER_H
#define NSAP_CLIQUEFINDER_H

#include <vector>
#include "Organism.h"
#include "Graph.h"

//#define NSAP_MODE_CPU0 //UNCOMMENT FOR CLASSIC CPU MODE
#define NSAP_MODE_GPU
/*
 * Class representing logic of algorithm
 * All organisms from population represent the same feat
 */
#ifdef NSAP_MODE_GPU
#include "../Headers/DeviceGraph.cuh"
#endif
class CliqueFinder {
public:
    void crossOver(std::vector<Organism> &pop, unsigned long childrenAmount);
    std::vector<int> randPerm(unsigned int size);

#if defined(NSAP_MODE_CPU0)
	int getWorth(Organism pop);
	int CliqueFinder::RyBKA(int sr, std::set<int> &p);
#endif
#ifdef NSAP_MODE_GPU
	DeviceGraph *dig;
#endif

    void selection(std::vector<Organism> &newPop);
    void nextGeneration();
    std::vector<Organism> population;
    Graph graph;
    int epoch = 0;
    int maxEpoch;
    int cliqueFeat = 0;
    double pMut = 0.4;

    std::pair<Organism, int> start();

    CliqueFinder(const Graph &g, const int startAmount, const unsigned int startSize, const int feat,
                 const int desMaxEpoch);
	~CliqueFinder();
};


#endif //NSAP_CLIQUEFINDER_H
