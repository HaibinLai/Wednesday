#include <iostream>
#include <cuda.h>

typedef int SIZE_TYPE; // 你可以根据需要定义 SIZE_TYPE

// 主函数
int main() {
    const int arraySize = 5; // 数组大小
    SIZE_TYPE host_num[arraySize] = {10, 20, 30, 40, 50}; // 主机上的数组
    SIZE_TYPE *node_queue; // 设备指针

    // 1. 在设备上分配内存
    cudaMalloc((void**)&node_queue, arraySize * sizeof(SIZE_TYPE));

    // 2. 将数据从主机复制到设备
    cudaMemcpy(node_queue, host_num, arraySize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);

    // 3. 创建一个主机数组以存储从设备复制的数据
    SIZE_TYPE *host_queue = new SIZE_TYPE[arraySize];

    // 4. 将设备数据复制回主机
    cudaMemcpy(host_queue, node_queue, arraySize * sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost);

    // 5. 打印主机上的数据
    std::cout << "node_queue values: ";
    for (int i = 0; i < arraySize; ++i) {
        std::cout << host_queue[i] << " "; // 打印每个值
    }
    std::cout << std::endl;

    // 6. 释放设备内存
    cudaFree(node_queue);
    delete[] host_queue; // 释放主机内存

    return 0;
}
