#include "../Headers/DeviceGraph.cuh"
#include "cuda_runtime.h"

__device__ int RyBKA(DeviceBitset *stack, int *map, int *rsstack, int N, const DeviceGraph *graph) {
	int stackIdx = 1, cmax = -1;
	while (stackIdx >= 0) {//while stack not empty
		stackIdx--; //stack pop
		if (stack[stackIdx].n == 0) {//if the P set is empty
			if (cmax < rsstack[stackIdx]) cmax = rsstack[stackIdx]; //check if found clique was greater than the previous one, if so, set
		}
		else {//if P is not empty
			int i = 0;//pick a vertex v
			while (!stack[stackIdx][i]) i++;//that exists in P
			for (int j = 0; j < N; j++) {//push (P \ v)
				if (stack[stackIdx][j] && j != i) stack[stackIdx + 1].set(i, 1);
				else stack[stackIdx + 1].set(i, 0);
			}
			rsstack[stackIdx + 1] = rsstack[stackIdx];
			stack[stackIdx + 1].n = stack[stackIdx].n - 1;
			stackIdx++;
			int m = 0;
			for (int j = 0; j < N; j++) {//for every other vertex
				if (stack[stackIdx][j]) {//that exists, do
					if (graph->isEdge(map[i], map[j])) {//check for edge by the way of map
						stack[stackIdx + 1].set(j, 1); //if connected, add to next iteration
						m++; //also, count
					}
					else stack[stackIdx + 1].set(j, 0); //if not, make sure it won't be there.
				}

			}
			stack[stackIdx + 1].n = m;
			rsstack[stackIdx + 1] = rsstack[stackIdx] + 1;
			stackIdx++;
		}
	}
	return cmax;
}

__device__ void getWorthDev(DeviceBKInput *in) {
	in->result = RyBKA(*in->set, in->map, in->rsstack, in->set[0]->n, in->g);
}
__host__ void indirectMalloc(void **ptr, int size) {
	void *p;
	cudaMalloc(&p, size);
	cudaMemcpy(ptr, &p, sizeof(void*), cudaMemcpyHostToDevice);
}

__host__ DeviceGraph* loadGraphToDevice(const Graph *g) {
	DeviceGraph *res;
	cudaMalloc((void**)&res, sizeof(DeviceGraph));
	indirectMalloc((void**)&res->vertices, g->vertexAmount*sizeof(DeviceGraphVertex));
	for (int i = 0; i < g->vertexAmount; i++) {
		DeviceGraphVertex v;
		v.neighbors = new int[g->vertices[i].neighbourhood.size()];
		int *neighbors;
		cudaMalloc((void**)&neighbors, g->vertices[i].neighbourhood.size()*sizeof(int));
		for (unsigned int j = 0; j < g->vertices[i].neighbourhood.size(); j++) {
			v.neighbors[j] = g->vertices[i].neighbourhood[j];
		}
		v.degree = g->vertices[i].neighbourhood.size();
		cudaMemcpy(neighbors, v.neighbors, g->vertices[i].neighbourhood.size()*sizeof(int), cudaMemcpyHostToDevice);
		delete[] v.neighbors;
		v.neighbors = neighbors;
		cudaMemcpy((&res->vertices)+i*sizeof(DeviceGraphVertex), &v, sizeof(DeviceGraphVertex), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(&res->n, &g->vertexAmount, sizeof(int), cudaMemcpyHostToDevice);
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
	indirectMalloc((void**)&res->contents, sizeof(char)*idx);
	for (int i = 0; i < idx; i++) res->contents[i] = (signed char)0xFF;
	cudaMemcpy(&res->n, &n, sizeof(int), cudaMemcpyHostToDevice);
	return res;
}

__host__ DeviceBitset** createBitsetArray(int n) {
	DeviceBitset **res;
	cudaMalloc((void**)&res, n*sizeof(DeviceBitset*));
	for (int i = 0; i < n; i++) {
		DeviceBitset *temp = createDeviceBitset(n);
		cudaMemcpy((&res) + i*sizeof(DeviceBitset*), &temp, sizeof(DeviceBitset*), cudaMemcpyHostToDevice);
	}
	return res;
}

__host__ void unloadDeviceGraph(DeviceGraph *g) {
	int n;
	cudaMemcpy(&g->n, &n, sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < n; i++) {
		DeviceGraphVertex v;
		cudaMemcpy(&g->vertices + i*sizeof(DeviceGraphVertex), &v, sizeof(DeviceGraphVertex), cudaMemcpyDeviceToHost);
		cudaFree(v.neighbors);
	}
	cudaFree(g);
}
__host__ void freeDeviceBitset(DeviceBitset *b) {
	cudaFree(b->contents);
	cudaFree(b);
}

__host__ void freeBitsetArray(DeviceBitset **arr, int n) {
	for (int i = 0; i < n; i++) cudaFree(arr+i*sizeof(DeviceBitset*));
	cudaFree(arr);
}

__global__ void getWorthCudaKernel(DeviceBKInput **roadmap) {
	int myId = threadIdx.x;
	DeviceBKInput *myInput = roadmap[myId];
	getWorthDev(myInput);
}

__host__ void getWorthWithCuda(std::vector<Organism> &pop, DeviceGraph *g) {
	DeviceBKInput **map, **hostCopy = new DeviceBKInput*[pop.size()];
	std::vector<void*> general, bitsets, results;
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

		int *resstack;
		cudaMalloc((void**)&resstack, pop[i].vertices.size()*sizeof(int));
		cudaMemset(resstack, 0, pop[i].vertices.size()*sizeof(int));
		cudaMemcpy(&in->rsstack, &resstack, sizeof(resstack), cudaMemcpyHostToDevice);

		DeviceBitset** t = createBitsetArray(pop[i].vertices.size());
		cudaMemcpy(&in->set, &t, sizeof(t), cudaMemcpyHostToDevice);

		bitsets.push_back(t);
		results.push_back(&in->result);

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
		cudaMemcpy(&temp, results[i], sizeof(int), cudaMemcpyDeviceToHost);
		pop[i].worth = temp;
		for (auto &t : bitsets) freeBitsetArray((DeviceBitset**)t, pop[i].vertices.size());
	}
	for (auto &t : general) cudaFree(t);
}