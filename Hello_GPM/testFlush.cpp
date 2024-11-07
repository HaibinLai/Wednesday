#include <iostream>
#include <chrono>
#include <thread>

int main() {
for (int i = 0; i < 5; ++i) {
// 模拟耗时操作
std::this_thread::sleep_for(std::chrono::seconds(1));
// 输出进度并立即刷新
std::cout << i + 1 << ", " << std::flush;
}
std::cout << "操作完成!" << std::endl;
return 0;
}