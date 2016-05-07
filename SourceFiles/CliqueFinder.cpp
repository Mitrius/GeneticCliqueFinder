#include "../Headers/CliqueFinder.h"
#include <algorithm>
#include <fstream>
#include <iostream>

Organism CliqueFinder::crossOver(Organism &a,Organism &b) {
    Organism child;
    //TODO crossing two organisms(gets half of unique vertices from one, and half from second)
    return child;
}
/*
 * Generates random vertex permutation with given size
 */
std::vector<int> CliqueFinder::randPerm(unsigned int size) {
    std::vector<int> perm;
    for(int i=0;i<graph->vertexAmount;i++){
        perm.push_back(i);
    }
    std::random_shuffle(perm.begin(),perm.end());
    perm.resize(size);
    return perm;
}
/*
 * Tournament selection of Organisms
 */
void CliqueFinder::selection(std::vector<Organism> currentPop,std::vector<Organism> newPop) {
    //TODO implementing  tournament selection
}
/*
 * Next step of algorithm, doing selection, mutations, crossing over and replaces population;
 */
void CliqueFinder::nextGeneration() {
    //No crossing at this moment
    std::vector<Organism> newPop;
    selection(population,newPop);
    for(int i=0;i<newPop.size();i++){
        double z = rand()%RAND_MAX;
        if(z < pMut){
            newPop[i].mutate();
        }
    }
    //TODO implement methods
    this->population = newPop;

}
int CliqueFinder::getWorth(std::vector<Organism> pop) {
    //TODO implement Bron-Kerbosch algorithm for clique number
    return 0;
}

CliqueFinder::CliqueFinder(const Graph &g,int startAmount,unsigned int startSize) {
    this->graph = &g;
    std::vector<int> perm;
    Organism tempOrg;
    for(int i=0;i<startAmount;i++){
        perm = randPerm((startSize));
        tempOrg.vertices.clear();
        for(int j = 0;j<perm.size();j++) {
            tempOrg.vertices.insert(perm[j]);
        }
        population.push_back(tempOrg);
    }
}


