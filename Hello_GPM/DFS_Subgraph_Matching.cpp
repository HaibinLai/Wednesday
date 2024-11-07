#include <iostream>
#include <vector>
#include <stack>

class Graph {
public:
    std::vector<std::vector<int>> adjList;

    // 构造函数
    Graph(int vertices) {
        adjList.resize(vertices);
    }

    // 添加边
    void addEdge(int u, int v) {
        adjList[u].push_back(v);
        adjList[v].push_back(u); // 无向图
    }

    // 非递归DFS遍历
    void DFS(int start) {
        std::vector<bool> visited(adjList.size(), false); // 访问标记
        std::stack<int> s; // 栈用于保存节点

        s.push(start); // 将起始节点推入栈

        while (!s.empty()) {
            int current = s.top(); // 获取栈顶元素
            s.pop(); // 弹出栈顶元素

            // 如果当前节点未被访问
            if (!visited[current]) {
                visited[current] = true; // 标记为已访问
                std::cout << current << " "; // 处理当前节点

                // 遍历邻接节点并将未访问的节点推入栈
                for (int neighbor : adjList[current]) {
                    if (!visited[neighbor]) {
                        s.push(neighbor);
                    }
                }
            }
        }
        std::cout << std::endl; // 输出结束后换行
    }
};

int main() {
    Graph g(5); // 创建一个包含5个顶点的图

    // 添加边
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 4);

    // 从顶点0开始DFS遍历
    std::cout << "非递归 DFS 遍历结果: ";
    g.DFS(0);

    return 0;
}

