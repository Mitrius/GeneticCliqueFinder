#include "../Headers/CliqueFinder.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cassert>

/*
 * Children gets all unique vertices from parents
 */
void CliqueFinder::crossOver(std::vector<Organism> &pop, const unsigned long childrenAmount) {
    Organism child;
    Organism father;
    Organism mother;
    unsigned long chosenOrganism;
    for (int i = 0; i < childrenAmount; i++) {
        child.vertices.clear();
        chosenOrganism = rand() % pop.size();
        father = pop[chosenOrganism];
        pop.erase(pop.begin() + chosenOrganism);//We don't support autogamy
        chosenOrganism = rand() % pop.size();
        mother = pop[chosenOrganism];
        child.vertices = father.vertices;
        child.vertices.insert(mother.vertices.begin(), mother.vertices.end());
        pop.push_back(father);
    }
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
void CliqueFinder::selection(std::vector<Organism> &currentPop, std::vector<Organism> &newPop) {
    newPop = currentPop;
}
/*
 * Next step of algorithm, doing selection, mutations, crossing over and replaces population;
 */
void CliqueFinder::nextGeneration() {

    std::vector<Organism> newPop;
    selection(population,newPop);
    crossOver(newPop, population.size() - newPop.size());
    for(int i=0;i<newPop.size();i++){
        double z = rand()%RAND_MAX;
        if(z < pMut){
            newPop[i].mutate(graph->vertexAmount);
        }
    }
    //TODO implement methods
    this->population = newPop;

}
int CliqueFinder::getWorth(std::vector<Organism> pop) {
    //TODO implement Bron-Kerbosch algorithm for clique number
    return 0;
}

CliqueFinder::CliqueFinder(const Graph &g, const int startAmount, const unsigned int startSize, const int feat,
                           const int maxEpoch) {
    graph = &g;
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
    cliqueFeat = feat;
    this->maxEpoch = maxEpoch;
}

/*
 * Main function of class, returning Best clique (organism, along with possible clique size);
 */
std::pair<Organism, int> CliqueFinder::start() {
    assert(epoch < maxEpoch);
    while (epoch < maxEpoch) {
        nextGeneration();
        epoch++;
    }
    std::sort(population.begin(), population.end());
    int possibleCliqueSize = 0;
    for (const auto setItem:population[0].vertices) {
        if (std::find(graph->vertices[setItem].feats.begin(), graph->vertices[setItem].feats.end(), cliqueFeat) !=
            graph->vertices[setItem].feats.end()) {
            possibleCliqueSize++;
        }
    }
    std::pair<Organism, int> retVal(population[0], possibleCliqueSize);
    return retVal;
}




