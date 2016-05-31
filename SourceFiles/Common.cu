#include "../Headers/Common.cuh"

__device__ int RyBKA(DeviceBitset *stack, int *map, int N, const DeviceGraph *graph) {
	int stackIdx = 1, rs = 0, cmax = -1;
	while (stackIdx>=0) {
		stackIdx--; //stack pop
		if (stack[stackIdx].n == 0) {//if the P set is empty
			if (cmax < rs) cmax = rs; //check if found clique was greater than the previous one, if so, set
		}
		else {//if P is not empty
			int i = 0;//pick a vertex v
			while (!stack[stackIdx][i]) i++;//that exists in P
			for (int j = 0; j < N; j++) {//push (P \ v)
				if (stack[stackIdx][j] && j != i) stack[stackIdx + 1].set(i, 1);
				else stack[stackIdx + 1].set(i, 0);
			}
			stack[stackIdx + 1].n = stack[stackIdx].n - 1;
			stackIdx++;
			int m = 0;
			for (int j = 0; j < N; j++) {//for every other vertex
				if (stack[stackIdx][j]) {//that exists, do
					if (graph->isEdge(map[i], map[j])) {//check for edge by the way of map
						stack[stackIdx + 1].set(j, 1); //if connected, add to next iteration
						m++; //also, count
					} else stack[stackIdx + 1].set(j, 0); //if not, make sure it won't be there.
				}

			}
			stackIdx++;
		}
	}
	return cmax;
}

__global__ void getWorthDev(DeviceBKInput *in) {
	in->result = RyBKA(*in->set, in->map, in->set[0]->n, in->g);
}