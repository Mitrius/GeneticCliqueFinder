#include "../Headers/DeviceGraph.cuh"
#include "cuda_runtime.h"

__host__ __device__ int RyBKA(DeviceBitset *stack, int *map, int *rsstack, int N, const DeviceGraph *graph) {
	int stackIdx = 1, cmax = -1;
	while (stackIdx > 0) {//while stack not empty
		stackIdx--; //stack pop
		if (stack[stackIdx].n == 0) {//if the P set is empty
			if (cmax < rsstack[stackIdx]) cmax = rsstack[stackIdx]; //check if found clique was greater than the previous one, if so, set
		}
		else {//if P is not empty
			int i = 0;//pick a vertex v
			while (!stack[stackIdx][i]) i++;//that exists in P
			//now to copy to unofficial N-th bitset
			for (int j = 0; j < (N / 8) + 1; j++) stack[N].contents[j] = stack[stackIdx].contents[j];
			rsstack[N] = rsstack[stackIdx];
			stack[N].n = stack[stackIdx].n;
			for (int j = 0; j < N; j++) {//push (P \ v)
				if (stack[N][j] && j != i) stack[stackIdx].set(i, 1);
				else stack[stackIdx].set(i, 0);
			}
			rsstack[stackIdx] = rsstack[N];
			stack[stackIdx].n = stack[N].n - 1;
			stackIdx++;
			int m = 0;
			for (int j = 0; j < N; j++) {//for every other vertex
				if (stack[N][j]) {//that exists, do
					if (graph->isEdge(map[i], map[j])) {//check for edge by the way of map
						stack[stackIdx].set(j, 1); //if connected, add to next iteration
						m++; //also, count
					}
					else stack[stackIdx].set(j, 0); //if not, make sure it won't be there.
				}

			}
			stack[stackIdx].n = m;
			rsstack[stackIdx] = rsstack[N] + 1;
			stackIdx++;
		}
	}
	return cmax;
}

__host__ DeviceGraph* loadGraphToDevice(const Graph *g) {
	DeviceGraph *res;
	cudaMallocManaged(&res, sizeof(DeviceGraph));
	res->n = g->vertexAmount;
	cudaMallocManaged(&res->vertices, g->vertexAmount*sizeof(DeviceGraphVertex));
	for (int i = 0; i < g->vertexAmount; i++) {
		int m = g->vertices[i].neighbourhood.size();
		res->vertices[i].degree = m;
		cudaMallocManaged(&res->vertices[i].neighbors, m*sizeof(int));
		for (unsigned int j = 0; j < m; j++) {
			res->vertices[i].neighbors[j] = g->vertices[i].neighbourhood[j];
		}
	}
	return res;
}

__host__ __device__ bool DeviceBitset::operator[](int n) {
	int idx = n / 8;
	int rem = n % 8;
	char mask = 0x1 << rem;
	return (contents[idx] & mask) ? true : false;
}

__host__ __device__ void DeviceBitset::set(int n, bool v) {
	int idx = n / 8;
	int rem = n % 8;
	char mask = 1 << rem;
	if (v) {
		if (contents[idx] != contents[idx] | mask) n++;
		contents[idx] = contents[idx] | mask;
	}
	else {
		if (contents[idx] != contents[idx] & ~mask) n--;
		contents[idx] = contents[idx] & ~mask;
	}
}

__host__ void unloadDeviceGraph(DeviceGraph *g) {
	for (int i = 0; i < g->n; i++) cudaFree(g->vertices[i].neighbors);
	cudaFree(g->vertices);
	cudaFree(g);
}

__host__ DeviceBitset* createBitsetArray(int n) {
	DeviceBitset *res;
	cudaMallocManaged(&res, (n+1)*sizeof(DeviceBitset));
	char *c;
	for (int i = 0; i < n+1; i++) {
		cudaMallocManaged(&c, (n / 8) + 1);
		for (int j = 0; j < (n / 8) + 1; j++) c[j] = 0xFF;
		res[i].contents = c;
		res[i].n = n;
	}
	return res;
}

__global__ void getWorthCudaKernel(DeviceBKInput **roadmap) {
	int myId = threadIdx.x;
	DeviceBKInput* in = roadmap[myId];
	in->result = RyBKA(in->set, in->map, in->rsstack, in->set[0].n, in->g);
}

__host__ void getWorthWithCuda(std::vector<Organism> &pop, DeviceGraph *g) {
	int N = pop.size();
	DeviceBKInput **roadmap;
	cudaMallocManaged(&roadmap, N*sizeof(DeviceBKInput*));
	for (int i = 0; i < N; i++) {
		int M = pop[i].vertices.size();
		DeviceBKInput *current;
		cudaMallocManaged(&current, sizeof(DeviceBKInput));
		current->result = -1;
		current->g = g;
		cudaMallocManaged(&current->map, (M+1)*sizeof(int));
		int j = 0;
		for (auto &t : pop[i].vertices) current->map[j++] = t;
		cudaMallocManaged(&current->rsstack, M*sizeof(int));
		for (int k = 0; k < M; k++) current->rsstack[k] = 0;
		current->set = createBitsetArray(M);
		roadmap[i] = current;
	}
	for (int i = 0; i < N; i++){
		DeviceBKInput* in = roadmap[i];	
		in->result = RyBKA(in->set, in->map, in->rsstack, in->set[0].n, in->g);
	}
	for (int i = 0; i < N; i++) {
		pop[i].worth = roadmap[i]->result;
		cudaFree(roadmap[i]->rsstack);
		cudaFree(roadmap[i]->map);
		for (int j = 0; j < roadmap[i]->set->n; j++) cudaFree(roadmap[i]->set[j].contents);
		cudaFree(roadmap[i]->set);
		cudaFree(roadmap[i]);
	}
	cudaFree(roadmap);
}