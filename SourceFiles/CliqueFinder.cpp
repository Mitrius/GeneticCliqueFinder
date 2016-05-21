#include "../Headers/CliqueFinder.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>

void CliqueFinder::crossOver(std::vector<Organism> &pop) {
    Organism child;
    //TODO crossing two organisms(gets half of unique vertices from one, and half from second)
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

    std::vector<Organism> newPop;
    selection(population,newPop);
    crossOver(newPop);
    for(int i=0;i<newPop.size();i++){
        double z = rand()%RAND_MAX;
        if(z < pMut){
            newPop[i].mutate(graph->vertexAmount);
        }
    }
    //TODO implement methods
    this->population = newPop;

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

CliqueFinder::CliqueFinder(const Graph &g, int startAmount, unsigned int startSize, int feat) {
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
}

std::vector<Organism> CliqueFinder::start() {
    std::vector<Organism> bestClique;

    return bestClique;
}




