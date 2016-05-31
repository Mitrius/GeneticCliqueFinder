//
// Created by Mitrius on 04.05.16.
//

#include "../Headers/Vertex.h"

Vertex::Vertex() {

}
bool Vertex::operator<(const Vertex &a) {
    return id > a.id;
}
bool Vertex::operator == (const Vertex &a){
	return id == a.id;
}



