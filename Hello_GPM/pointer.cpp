#include <iostream>

// 函数声明
void add(int a, int b, int* result);

int main() {
    int x = 5;          // 第一个加数
    int y = 10;         // 第二个加数
    int sum;           // 用于存储结果

    // 调用加法函数
    add(x, y, &sum);

    // 输出结果
    std::cout << "The result returned to main is: " << sum << std::endl;

    return 0;
}

// 加法函数实现
void add(int a, int b, int* result) {
    *result = a + b;   // 将加法结果存储在 result 指向的内存地址
    std::cout << "Inside add function: " << a << " + " << b << " = " << *result << std::endl; // 打印加法结果
}
