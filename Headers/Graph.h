#ifndef NSAP_GRAPH_H
#define NSAP_GRAPH_H

#include <vector>
#include <string>
#include "Vertex.h"
#include <set>

class Graph {
public:
    Graph(std::string fileName);
	std::vector<int> IdMap;
    Graph();
    int vertexAmount;
    std::vector<Vertex> vertices;
    std::vector<std::string> featDescriptorArray;
private:
    void combineGraph(const std::vector<std::vector<std::string>> &edgeList, const std::vector <std::string> &idArray,
                      const std::vector<std::vector<bool, std::allocator<bool>>> &feats);
};


#endif //NSAP_GRAPH_H
