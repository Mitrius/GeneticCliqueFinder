#include <fstream>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <ctime>

#include "../Headers/Graph.h"
void Graph::combineGraph(const std::vector<std::vector <std::string> > &edgeList,const std::vector<std::string> &idArray
,const std::vector<std::vector<bool> > &feats ){
    std::string currentId;
    Vertex tempVert;
    assert(vertexAmount == feats.size()); //Sanity check
    for(unsigned int i=0;i<(unsigned int)vertexAmount;i++){
        tempVert.id = i;
        tempVert.feats.clear();
        for(unsigned int j=0;j<feats[i].size();j++){
            assert(i<feats.size());
            assert(j<feats[i].size());
            if(feats[i][j] == 1)
                tempVert.feats.push_back(j);
        }
        vertices.push_back(tempVert);
    }
    for(unsigned int i=0;i<vertices.size();i++){
        currentId = idArray[i];
        for(unsigned int j = 0; j<edgeList.size();j++){
            if(edgeList[j][0].compare(currentId) == 0){
                for(unsigned int k =0;k<idArray.size();k++){
                    if(idArray[k].compare(edgeList[j][1]) == 0){
                        vertices[i].neighbourhood.push_back(k);
                        break;
                    }
                }
            }
        }
    }

}

Graph::Graph(const std::string fileName) {
    std::ifstream inputStream;
    std::vector<std::vector<std::string> > edgeList;
    std::vector<std::vector<bool> > feats;
    std::vector<std::string> idArray;
    std::string truncFilename;
    time_t finishTime = 0;
    time_t start;
    /*
    * Loads feats from filename.feats file feat look like that:
    * vertex feat1(1/0) feat2(1/0) ... \n
    */
    start = time(0);
    truncFilename = fileName;
    truncFilename += ".feat";
    inputStream.open(truncFilename.c_str());
    std::string line;
    std::string id;
    std::vector<bool> empty;
    assert(inputStream.good());
    while (inputStream.peek() != EOF) {
        empty.clear();
        id.clear();

        std::getline(inputStream, line, '\n');//'\n' przy getline to troche overkill ;)
        unsigned long i = 0;

        while (line[i] != ' ') {//extracting ID
            id += line[i];
            i++;
        }

        idArray.push_back(id);
        while (i < line.size()) {
            if (line[i] != ' ')
                empty.push_back((line[i] - 48) == 1);// 1/0, feat order persists
            i++;
        }
        feats.push_back(empty);
    }
    inputStream.close();
    truncFilename = fileName;
    truncFilename += ".edges";
    vertexAmount = (int) feats.size();
    /*
     * Load edges from filename.edges file Edge looks like that:
     * vertex1 vertex2
     * graph is directed!
     */
    inputStream.open(truncFilename.c_str());
    assert(inputStream.good());
    line = "";
    std::vector<std::string> edge(2);
    while (inputStream.peek() != EOF) {
        std::getline(inputStream, line, '\n');
        unsigned int i = 0;
        edge[0] = "";
        edge[1] = "";
        while (line[i] != ' ') {
            edge[0] += line[i];
            i++;
        }
        i++;
        while (i < line.size()) {
            edge[1] += line[i];
            i++;
        }
        edgeList.push_back(edge);
    }
    inputStream.close();
    combineGraph(edgeList, idArray, feats);
    /*
     * Load description of feats
     */
    truncFilename = fileName;
    truncFilename += ".featnames";
    inputStream.open(truncFilename.c_str());
    assert(inputStream.good());
    line = "";
    std::string inputString;
    unsigned int i = 0;
    while (inputStream.peek() != EOF) {
        i = 0;
        inputString = "";
        std::getline(inputStream, line, '\n');
        while (line[i] != ' ') {//skip shit
            i++;
        }
        for (; i < line.length(); i++) {
            inputString += line[i];
        }
        featDescriptorArray.push_back(inputString);
    }
    finishTime = time(0);
    inputStream.close();
    std::cout << "Graph has been created with: " << vertexAmount << " vertices, in: " << difftime(finishTime, start) <<
    "seconds" << std::endl;

}

Graph::Graph() {

}





