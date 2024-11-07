#include <iostream>
#include <cuda_runtime.h>

__global__ void findUnique(int *input, int *unique, int *uniqueCount, int numElements) {
    // 每个线程处理一个元素
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // 使用共享内存存储唯一值
    extern __shared__ int sharedUnique[];

    if (idx < numElements) {
        int value = input[idx];
        bool isUnique = true;

        // 检查该值是否已经在 sharedUnique 中
        for (int j = 0; j < *uniqueCount; j++) {
            if (sharedUnique[j] == value) {
                isUnique = false;
                break;
            }
        }

        // 如果该值是唯一的，添加到 sharedUnique 中
        if (isUnique) {
            int pos = atomicAdd(uniqueCount, 1);
            sharedUnique[pos] = value;
        }
    }

    // 等待所有线程完成
    __syncthreads();

    // 将 sharedUnique 的结果写回 global memory
    if (threadIdx.x == 0) {
        for (int j = 0; j < *uniqueCount; j++) {
            unique[j] = sharedUnique[j];
        }
    }
}

int main() {
    const int numElements = 10; // 数组大小
    int h_input[numElements] = {1, 2, 3, 4, 2, 3, 5, 1, 6, 7}; // 示例输入
    int *d_input, *d_unique, *d_uniqueCount;
    int h_uniqueCount = 0;

    // 分配设备内存
    cudaMalloc((void**)&d_input, numElements * sizeof(int));
    cudaMalloc((void**)&d_unique, numElements * sizeof(int)); // 最多存储 numElements 个唯一值
    cudaMalloc((void**)&d_uniqueCount, sizeof(int));

    // 将输入数据复制到设备
    cudaMemcpy(d_input, h_input, numElements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uniqueCount, &h_uniqueCount, sizeof(int), cudaMemcpyHostToDevice);

    // 启动 kernel
    findUnique<<<(numElements + 255) / 256, 256, numElements * sizeof(int)>>>(d_input, d_unique, d_uniqueCount, numElements);

    // 从设备复制回 uniqueCount 和 unique 数组
    cudaMemcpy(&h_uniqueCount, d_uniqueCount, sizeof(int), cudaMemcpyDeviceToHost);
    int *h_unique = new int[h_uniqueCount]; // 在主机上创建唯一数组
    cudaMemcpy(h_unique, d_unique, h_uniqueCount * sizeof(int), cudaMemcpyDeviceToHost);

    // 输出结果
    std::cout << "Unique values: ";
    for (int i = 0; i < h_uniqueCount; i++) {
        std::cout << h_unique[i] << " ";
    }
    std::cout << std::endl;

    // 清理
    delete[] h_unique;
    cudaFree(d_input);
    cudaFree(d_unique);
    cudaFree(d_uniqueCount);

    return 0;
}
