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
	in->result = RyBKA(in->set, in->map, in->rsstack, in->set[0].n, in->g);
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

__host__ DeviceBitset* createBitsetArray(int n) {
	DeviceBitset *res, model, *temp;
	cudaMalloc(&res, n*sizeof(DeviceBitset));
	model.n = n;
	temp = res;
	char *c;
	for (int i = 0; i < n; i++) {
		cudaMalloc(&c, (n / 8) + 1);
		cudaMemset(c, 0xFF, (n / 8) + 1);
		model.contents = c;
		cudaMemcpy(temp, &model, sizeof(model), cudaMemcpyHostToDevice);
		temp += sizeof(model);
	}
	return res;
}

__global__ void getWorthCudaKernel(DeviceBKInput **roadmap) {
	int myId = threadIdx.x;
	DeviceBKInput* myInput = roadmap[myId];
	getWorthDev(myInput);
}

__host__ void getWorthWithCuda(std::vector<Organism> &pop, DeviceGraph *g) {
	int N = pop.size();
	DeviceBKInput **roadmap = new DeviceBKInput*[N], **newroadmap = new DeviceBKInput*[N];
	for (int i = 0; i < N; i++) {
		int M = pop[i].vertices.size();
		roadmap[i] = new DeviceBKInput;
		roadmap[i]->g = g;
		roadmap[i]->result = -1;
		int *rsstack;
		cudaMalloc(&rsstack, sizeof(int)*M);
		roadmap[i]->rsstack = rsstack;
		int *map, *mapH = new int[M], it = 0;
		cudaMalloc(&map, sizeof(int)*M);
		for (auto &t : pop[i].vertices) mapH[it++] = t;
		cudaMemcpy(map, mapH, sizeof(int)*M, cudaMemcpyHostToDevice);
		roadmap[i]->set = createBitsetArray(M);
		DeviceBKInput *t;
		cudaMalloc(&t, sizeof(DeviceBKInput));
		cudaMemcpy(t, roadmap[i], sizeof(DeviceBKInput), cudaMemcpyHostToDevice);
		newroadmap[i] = t;
		delete[] mapH;
	}
	DeviceBKInput **tab;
	cudaMalloc(&tab, sizeof(DeviceBKInput*)*N);
	cudaMemcpy(tab, newroadmap, sizeof(DeviceBKInput*)*N, cudaMemcpyHostToDevice);

	getWorthCudaKernel<<<1, N>>>(roadmap);
	cudaDeviceSynchronize();	//must remember this, or bad things will happen. baad thing
								//cuda kernel launched and finished
	for (int i = 0; i < N; i++) {
		DeviceBKInput bi;
		cudaMemcpy(&bi, tab+i*sizeof(bi), sizeof(bi), cudaMemcpyDeviceToHost);
		pop[i].worth = bi.result;
		cudaFree(roadmap[i]->rsstack);
		cudaFree(roadmap[i]->set);
		cudaFree(roadmap[i]->map);
		cudaFree(newroadmap[i]);
	}
	cudaFree(tab);
	for (int i = 0; i < N; i++) delete roadmap[i];
	delete[] roadmap;
	delete[] newroadmap;
}