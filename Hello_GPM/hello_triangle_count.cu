#include <iostream>
#include <vector>
#include <cuda.h>

using namespace std;

__global__ void checkTriangles(int* adjMatrix, int numNodes, bool* found) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node < numNodes) {
        for (int neighbor = 0; neighbor < numNodes; neighbor++) {
            if (adjMatrix[node * numNodes + neighbor] == 1) { // 检查邻接
                for (int secondNeighbor = neighbor + 1; secondNeighbor < numNodes; secondNeighbor++) {
                    // 检查是否形成三角形
                    if (adjMatrix[neighbor * numNodes + secondNeighbor] == 1 && 
                        adjMatrix[node * numNodes + secondNeighbor] == 1) {
                        *found = true; // 找到三角形
                    }
                }
            }
        }
    }
}

int main() {
    // 定义图的节点数
    const int numNodes = 5;

    // 定义邻接矩阵
    int h_adjMatrix[numNodes][numNodes] = {
        {0, 1, 1, 0, 0},
        {1, 0, 1, 1, 0},
        {1, 1, 0, 0, 0},
        {0, 1, 0, 0, 1},
        {0, 0, 0, 1, 0}
    };

    // 分配设备内存
    int* d_adjMatrix;
    bool* d_found;
    bool h_found = false;

    cudaMalloc(&d_adjMatrix, numNodes * numNodes * sizeof(int));
    cudaMalloc(&d_found, sizeof(bool));

    // 拷贝邻接矩阵到设备
    cudaMemcpy(d_adjMatrix, h_adjMatrix, numNodes * numNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_found, &h_found, sizeof(bool), cudaMemcpyHostToDevice);

    // 启动核函数，使用 numNodes 个线程
    checkTriangles<<<(numNodes + 255) / 256, 256>>>(d_adjMatrix, numNodes, d_found);

    // 拷贝结果回主机
    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

    if (h_found) {
        cout << "图中存在三角形\n";
    } else {
        cout << "图中不存在三角形\n";
    }

    // 释放设备内存
    cudaFree(d_adjMatrix);
    cudaFree(d_found);

    return 0;
}
