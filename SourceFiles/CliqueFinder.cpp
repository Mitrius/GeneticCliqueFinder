#include "../Headers/CliqueFinder.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <cassert>

/*
 * Children gets all unique vertices from parents
 */
void CliqueFinder::crossOver(std::vector<Organism> &pop, const unsigned long childrenAmount) {
    Organism child;
    Organism father;
    Organism mother;
    unsigned long chosenOrganism;
    assert(pop.size() > 0);
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
        pop.push_back(child);//They grow soo fast after all
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
void CliqueFinder::selection(std::vector<Organism> &newPop) {
    unsigned long TournAmount = newPop.size() / 2;
    unsigned long cont1 = 0;
    unsigned long cont2 = 0;
    Organism tempOrg;
    for (int i = 0; i < TournAmount; i++) {
        cont1 = rand() % newPop.size();
        tempOrg = newPop[cont1];
        newPop.erase(newPop.begin() + cont1);
        cont2 = rand() % newPop.size();
        if (tempOrg.worth > newPop[cont2].worth) {
            newPop.erase(newPop.begin() + cont2);
            newPop.push_back(tempOrg);
        }
    }
}
/*
 * Next step of algorithm, doing selection, mutations, crossing over and replaces population;
 */
void CliqueFinder::nextGeneration() {
    std::vector<Organism> newPop;
    newPop = population;
    for (auto f:newPop)
        f.worth = getWorth(f);
    selection(newPop);
    crossOver(newPop, population.size() - newPop.size());
    for(int i=0;i<newPop.size();i++){
        double z = rand()%RAND_MAX;
        if(z < pMut){
            newPop[i].mutate(graph->vertexAmount);
        }
    }
    population = newPop;
}

int CliqueFinder::getWorth(Organism pop) {
    std::set<int> x, na = pop.vertices, r;
    return RyBKA(r, na, x);
}
int CliqueFinder::RyBKA(std::set<int> &r, std::set<int> &p, std::set<int> &x) {
    if (p.size() == 0 && x.size() == 0) return r.size();
    int cmax = -1;
    std::set<int> nr, np, nx;
    for (auto &t : p) {
        nr = r;
        np.clear();
        nx.clear();
        nr.insert(t);
        for (auto &v : p) {
            if (graph->isEdge(t, v, cliqueFeat) && graph->isEdge(v, t, cliqueFeat)) np.insert(v);
        }
        for (auto &v : x) {
            if (graph->isEdge(t, v, cliqueFeat) && graph->isEdge(v, t, cliqueFeat)) np.insert(v);
        }
        int temp = RyBKA(nr, np, nx);
        if (cmax < temp) cmax = temp;
    }
    return cmax;
}

CliqueFinder::CliqueFinder(const Graph &g, const int startAmount, const unsigned int startSize, const int feat,
                           const int desMaxEpoch) {
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
    maxEpoch = desMaxEpoch;
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




