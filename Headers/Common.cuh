#ifndef C_COMMON_H
#define C_COMMON_H

#include "thrust/device_vector.h"
#include "../Headers/DeviceGraph.cuh"


__device__ int RyBKA(DeviceBitset *stack, int *map, int N, const DeviceGraph *graph);
__global__ void getWorthDev(DeviceBKInput *in);

#endif