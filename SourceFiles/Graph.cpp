#include <fstream>
#include <cassert>
#include <iostream>
#include "../Headers/Graph.h"
void Graph::combineGraph(const std::vector<std::vector <std::string> > &edgeList,const std::vector<std::string> &idArray
,const std::vector<std::vector<bool> > &feats ){
    std::string currentId;
    Vertex tempVert;
    assert(vertexAmount == feats.size());
    for(int i=0;i<vertexAmount;i++){
        tempVert.id = i;
        tempVert.feats.clear();
        for(int j=0;j<feats[i].size();j++){
            assert(i<feats.size());
            assert(j<feats[i].size());
            if(feats[i][j] == 1)
                tempVert.feats.push_back(j);
        }
        vertices.push_back(tempVert);
    }
    for(int i=0;i<vertices.size();i++){
        currentId = idArray[i];
        for(int j = 0; j<edgeList.size();j++){
            if(edgeList[j][0].compare(currentId) == 0){
                for(int k =0;k<idArray.size();k++){
                    if(idArray[k].compare(edgeList[j][1]) == 0){
                        vertices[i].neighbourhood.push_back(k);
                        break;
                    }
                }
            }
        }
    }

}
Graph::Graph(const std::string fileName){
    std::ifstream inputStream;
    std::vector<std::vector<std::string> > edgeList;
    std::vector<std::vector<bool > > feats;
    std::vector<std::string> idArray;
    std::string truncFilename;
    /*
    * Load feats from filename.feats file feat look like that:
    * vertex feat1(1/0) feat2(1/0) ... \n
    */
    truncFilename  = fileName;
    truncFilename+=  ".feat";
    inputStream.open(truncFilename.c_str());
    int feat = 0;
    std::string line;
    std::string id;
    std::vector<bool> empty;
    assert(inputStream.good());
    while(inputStream.peek()!= EOF) {
        empty.clear();
        id.clear();

        std::getline(inputStream, line, '\n');//'\n' przy getline to troche overkill ;)
        unsigned long i = 0;

        while (line[i] != ' '){//extracting ID
            id += line[i];
            i++;
        }

        idArray.push_back(id);
        while(i < line.size()){
                if (line[i] != ' ')
                empty.push_back((bool) (line[i] - 48));// 1/0, feat order persists
                i++;
        }
        feats.push_back(empty);
    }
    inputStream.close();
    truncFilename  = fileName;
    truncFilename+=  ".edges";
    vertexAmount = (int) feats.size();
    /*
     * Load edges from filename.edges file Edge looks like that:
     * vertex1 vertex2
     * graph is directed!
     */
    inputStream.open(truncFilename.c_str());
    assert(inputStream.good());
    std::vector<std::string> edge(2);
    while(inputStream.peek()!= EOF){
        inputStream >> edge[0];
        inputStream >> edge[1];
        edgeList.push_back(edge);
    }
    inputStream.close();
    combineGraph(edgeList,idArray,feats);
    /*
     * Load description of feats
     */
    truncFilename = fileName;
    truncFilename += ".featnames";
    inputStream.open(truncFilename.c_str());
    assert(inputStream.good());
    line = "";
    std::string inputString;
    int i = 0;
    while(inputStream.peek() != EOF){
        i = 0;
        inputString = "";
        std::getline(inputStream,line,'\n');
        while(line[i] != ' '){//skip shit
            i++;
        }
        for(;i<line.length();i++){
            inputString += line[i];
        }
        featDescriptorArray.push_back(inputString);
    }
    inputStream.close();

}


