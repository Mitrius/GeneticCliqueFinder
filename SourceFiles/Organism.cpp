//
// Created by Mitrius on 22.04.16.
//

#include <cstdlib>
#include "../Headers/Organism.h"
/*
 * Mutating the organism (it gets one new vertex) or one of his vertices gets replaced
 * Value "TRUE" of muType indicates new vertex
 */
void Organism::mutate() {
        bool muType = (bool) (rand() % 2);
        if(muType){
            //TODO adding another vertex
        }
        else{
            //TODO replacing one of the vertices
        }
}




