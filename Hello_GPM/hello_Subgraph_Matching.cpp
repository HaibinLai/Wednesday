#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm> // 包含 std::find

using namespace std;

class Graph {
public:
    // 邻接表表示图
    unordered_map<int, vector<int>> adjList;

    // 添加边
    void addEdge(int u, int v) {
        adjList[u].push_back(v);
        adjList[v].push_back(u); // 无向图
    }
};

bool isSubgraphMatch(const Graph& G, const Graph& H, int vG, int vH, vector<int>& mapping, vector<bool>& visited) {
    // 如果所有的子图顶点都已匹配，返回真
    if (vH == H.adjList.size()) {
        return true;
    }

    // 尝试匹配大图G的每个未访问的顶点
    for (int u = 0; u < G.adjList.size(); ++u) {
        // 如果该顶点未被访问
        if (!visited[u]) {
            mapping[vH] = u; // 将当前G的顶点映射到H的顶点
            visited[u] = true;

            // 检查匹配的边是否存在
            bool isValid = true;
            for (int neighbor : H.adjList.at(u)) {
                if (mapping[neighbor] != -1 && 
                    find(G.adjList.at(u).begin(), G.adjList.at(u).end(), mapping[neighbor]) == G.adjList.at(u).end()) {
                    isValid = false;
                    break;
                }
            }

            // 如果当前匹配有效，递归查找下一个子图顶点
            // 这是最大的问题
            if (isValid && isSubgraphMatch(G, H, vG, vH + 1, mapping, visited)) {
                return true;
            }

            // 回溯
            visited[u] = false;
            mapping[vH] = -1; // 清空映射
        }
    }

    return false;
}

bool subgraphMatch(const Graph& G, const Graph& H) {
    vector<int> mapping(H.adjList.size(), -1); // 用于存储映射
    vector<bool> visited(G.adjList.size(), false); // 访问标记
    return isSubgraphMatch(G, H, G.adjList.size(), 0, mapping, visited);
}

int main() {
    Graph G, H;

    // 构建大图G
    G.addEdge(0, 1);
    G.addEdge(0, 2);
    G.addEdge(1, 2);
    G.addEdge(1, 3);
    G.addEdge(2, 4);

    // 构建子图H
    H.addEdge(0, 1);
    H.addEdge(1, 2);

    // 查找子图匹配
    if (subgraphMatch(G, H)) {
        cout << "子图匹配成功!" << endl;
    } else {
        cout << "子图匹配失败!" << endl;
    }

    return 0;
}
