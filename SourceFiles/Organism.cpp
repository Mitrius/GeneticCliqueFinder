//
// Created by Mitrius on 22.04.16.
//

#include <cstdlib>
#include "../Headers/Organism.h"
/*
 * Mutating the organism (it gets one new vertex) or one of his vertices gets replaced
 * Value "TRUE" of muType indicates new vertex
 */
void Organism::mutate(int vertexAmount) {
        bool muType = (bool) (rand() % 2);
        if(muType){
            int sizePre = (int) vertices.size();
            int newVert;
            while (vertices.size() == sizePre) {
                newVert = rand() % vertexAmount;
                vertices.insert(newVert);
            }
        }
        else{
            int sizePre = (int) vertices.size();
            int replaced = (int) (rand() % vertices.size());
            int replacing;
            while (vertices.size() == sizePre) {
                replacing = rand() % vertexAmount;
                vertices.insert(replacing);
            }
            vertices.erase(replaced);
        }
}




