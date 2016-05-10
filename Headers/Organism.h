//
// Created by Mitrius on 22.04.16.
//

#ifndef NSAP_ORGANISM_H
#define NSAP_ORGANISM_H

#include <vector>
#include <set>
#include "Graph.h"

/* Object representing clique in algorithm
    consists of:
        - set of vertices representing clique approximation(vertices are represented by their ID).
        - procedure mutating the clique(adding new vertex/replacing one)
 */
class Organism {
public:
    void mutate(int vertexAmount);
    std::set<int> vertices;
private:

};


#endif //NSAP_ORGANISM_H
