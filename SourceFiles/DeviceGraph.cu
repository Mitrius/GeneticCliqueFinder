#include "../Headers/DeviceGraph.cuh"

__host__ DeviceGraph* loadGraphToDevice(const Graph *g) {
	DeviceGraph *res;
	cudaMalloc((void**)&res, sizeof(DeviceGraph));
	cudaMalloc((void**)&res->vertices, g->vertexAmount*sizeof(DeviceGraphVertex));
	for (int i = 0; i < g->vertexAmount; i++) {
		cudaMalloc((void**)&res->vertices[i].neighbors, g->vertices[i].neighbourhood.size()*sizeof(int));
		for (int j = 0; j < g->vertices[i].neighbourhood.size(); j++) {
			res->vertices[i].neighbors[j] = g->vertices[i].neighbourhood[j];
		}
		res->vertices[i].degree = g->vertices[i].neighbourhood.size();
	}
	res->n = g->vertexAmount;
	return res;
}

__device__ bool DeviceBitset::operator[](int n) {
	int idx = n / 8;
	int rem = n % 8;
	char mask = 0x1 << rem;
	return (contents[idx] & mask) ? true : false;
}

__device__ void DeviceBitset::set(int n, bool v) {
	int idx = n / 8;
	int rem = n % 8;
	char mask = (v ? 1 : 0) << rem;
	contents[idx] = contents[idx] | mask;
}

__host__ DeviceBitset* createDeviceBitset(int n) {
	DeviceBitset *res;
	cudaMalloc((void**)&res, sizeof(DeviceBitset));
	int idx = n / 8;
	cudaMalloc((void**)&res->contents, sizeof(char)*idx);
	for (int i = 0; i < idx; i++) res->contents[i] = 0x0;
	res->n = n;
	return res;
}

__host__ void unloadDeviceGraph(DeviceGraph *g) {
	for (int i = 0; i < g->n; i++)
		cudaFree(g->vertices[i].neighbors);
	cudaFree(g);
}
__host__ void freeDeviceBitset(DeviceBitset *b) {
	cudaFree(b->contents);
	cudaFree(b);
}