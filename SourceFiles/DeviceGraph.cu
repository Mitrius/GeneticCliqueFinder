#include "../Headers/DeviceGraph.cuh"
#include "cuda_runtime.h"
#include "../Headers/Common.cuh"

__host__ DeviceGraph* loadGraphToDevice(const Graph *g) {
	DeviceGraph *res;
	cudaMalloc((void**)&res, sizeof(DeviceGraph));
	cudaMalloc((void**)&res->vertices, g->vertexAmount*sizeof(DeviceGraphVertex));
	for (int i = 0; i < g->vertexAmount; i++) {
		cudaMalloc((void**)&res->vertices[i].neighbors, g->vertices[i].neighbourhood.size()*sizeof(int));
		for (unsigned int j = 0; j < g->vertices[i].neighbourhood.size(); j++) {
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
	for (int i = 0; i < idx; i++) res->contents[i] = (signed char)0xFF;
	res->n = n;
	return res;
}

__host__ DeviceBitset** createBitsetArray(int n) {
	DeviceBitset **res;
	cudaMalloc((void**)&res, n*sizeof(DeviceBitset*));
	for (int i = 0; i < n; i++) res[i] = createDeviceBitset(n);
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

__host__ void freeBitsetArray(DeviceBitset **arr, int n) {
	for (int i = 0; i < n; i++) cudaFree(arr[i]);
	cudaFree(arr);
}

__global__ void getWorthCudaKernel(DeviceBKInput **roadmap) {
	int myId = threadIdx.x;
	DeviceBKInput *myInput = roadmap[myId];
	getWorthDev(myInput);
}

__host__ void getWorthWithCuda(std::vector<Organism> &pop, DeviceGraph *g) {
	DeviceBKInput **map, **hostCopy = new DeviceBKInput*[pop.size()];
	std::vector<void*> general, bitsets;
	cudaMalloc((void**)&map, sizeof(DeviceBKInput*)*pop.size());
	general.push_back(map);
	for (unsigned int i = 0; i < pop.size(); i++) {
		DeviceBKInput *in;
		cudaMalloc((void**)&in, sizeof(DeviceBKInput));
		general.push_back(in);
		int *vertexMap;
		cudaMalloc((void**)&vertexMap, pop[i].vertices.size()*sizeof(int));
		general.push_back(vertexMap);
		int *tempArray = new int[pop[i].vertices.size()], j=0;
		for (auto &t : pop[i].vertices) tempArray[j++] = t;
		cudaMemcpy(vertexMap, tempArray, sizeof(int)*pop[i].vertices.size(), cudaMemcpyHostToDevice);
		delete[] tempArray;
		cudaMemcpy(&in->map, &vertexMap, sizeof(vertexMap), cudaMemcpyHostToDevice);
		DeviceBitset** t = createBitsetArray(pop[i].vertices.size());
		bitsets.push_back(t);
		cudaMemcpy(&in->set, &t, sizeof(t), cudaMemcpyHostToDevice);
		cudaMemcpy(&in->g, &g, sizeof(g), cudaMemcpyHostToDevice);
		cudaMemcpy(&map[i], &in, sizeof(in), cudaMemcpyHostToDevice);
	}
	//input array ready
	//time to launch CUDA kernel
	int n = pop.size();
	getWorthCudaKernel<<<1, n>>>(map);
	cudaDeviceSynchronize(); //must remember this, or bad things will happen. baad things
	//cuda kernel launched and finished
	for (int i = 0; i < pop.size(); i++) {
		int temp;
		cudaMemcpy(&temp, &map[i]->result, sizeof(int), cudaMemcpyDeviceToHost);
		pop[i].worth = temp;
		for (auto &t : bitsets) freeBitsetArray((DeviceBitset**)t, pop[i].vertices.size());
	}
	for (auto &t : general) cudaFree(t);
}