#ifndef NSAP_DEVICE_GRAPH
#define NSAP_DEVICE_GRAPH
#include "thrust\device_vector.h"
#include "Graph.h"
#include <vector>
#include "Organism.h"
//#define NSAP_MODE_CPU0 //UNCOMMENT FOR CLASSIC CPU MODE
#define NSAP_MODE_GPU

struct DeviceGraphVertex {
	int *neighbors, degree;
};

struct DeviceGraph {
	int n;
	DeviceGraphVertex *vertices;
	__host__ __device__ bool isEdge(int v, int t) const {
		for (int i = 0; i < vertices[v].degree; i++) {
			if (vertices[v].neighbors[i] == t) return true;
		}
		return false;
	}
};

struct DeviceBitset {
	int n;
	char *contents;
	__host__ __device__ bool operator[](int n);
	__host__ __device__ void set(int n, char v);
};

struct DeviceBKInput {
	DeviceBitset *set;
	DeviceGraph *g;
	int *map;
	int *rsstack;
	int result;
};

__host__ DeviceGraph* loadGraphToDevice(const Graph *g);
__host__ void unloadDeviceGraph(DeviceGraph *g);
__host__ void getWorthWithCuda(std::vector<Organism> &pop, DeviceGraph *g);

#endif