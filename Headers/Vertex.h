//
// Created by Mitrius on 04.05.16.
//

#ifndef NSAP_VERTEX_H
#define NSAP_VERTEX_H
#include <vector>

class Vertex {
public:
    int id;
    std::vector<int> neighbourhood;
    std::vector<int> feats;
    Vertex();
    bool operator < (const Vertex &a);
	bool operator == (const Vertex &a);
};


#endif //NSAP_VERTEX_H
