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
        - set of vertices representing clique approximation.
 */

class Organism {
public:
    void mutate();
    std::set<int> vertices;
private:

};


#endif //NSAP_ORGANISM_H
