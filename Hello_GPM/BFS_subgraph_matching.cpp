#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>

using namespace std;

// 图类定义
class Graph {
public:
    // 使用邻接表表示图
    unordered_map<int, vector<int>> adjacencyList;

    // 添加边
    void addEdge(int u, int v) {
        adjacencyList[u].push_back(v);
        adjacencyList[v].push_back(u); // 无向图
    }

    // BFS子图匹配
    bool bfsSubgraphMatching(const vector<int>& subgraph, int startNode) {
        queue<int> q;
        unordered_set<int> visited;
        unordered_map<int, int> matching; // 子图节点到大图节点的匹配

        q.push(startNode);

        while (!q.empty()) {
            int current = q.front();
            q.pop();
            visited.insert(current);

            // 尝试将当前节点匹配到子图节点
            for (int subNode : subgraph) {
                if (matching.find(subNode) == matching.end()) {
                    matching[subNode] = current;
                    break;
                }
            }

            // 遍历当前节点的邻居
            for (int neighbor : adjacencyList[current]) {
                if (visited.find(neighbor) == visited.end()) {
                    q.push(neighbor);
                }
            }
        }

        // 检查是否所有子图节点都有匹配
        for (int subNode : subgraph) {
            if (matching.find(subNode) == matching.end()) {
                return false; // 失败，未找到完整匹配
            }
        }

        cout << "matching" << endl;
        // print match:
        for (auto it = matching.begin(); it != matching.end(); it++) {
            cout << it->first << " " << it->second << endl;
        }

        return true; // 成功，找到了匹配
    }
};

int main() {
    // 创建大图
    Graph mainGraph;
    mainGraph.addEdge(0, 1);
    mainGraph.addEdge(0, 2);
    mainGraph.addEdge(1, 3);
    mainGraph.addEdge(2, 3);
    mainGraph.addEdge(3, 4);

    // print maingraph:
    for (auto it = mainGraph.adjacencyList.begin(); it != mainGraph.adjacencyList.end(); it++) {
        cout << it->first << " ";
        for (int i = 0; i < it->second.size(); i++) {
            cout << it->second[i] << " ";
        }
        cout << endl;
    }

    // 定义子图
    vector<int> subgraph = {1, 2, 3}; // 要匹配的节点集合

    // 尝试从大图的某个起始节点查找子图的匹配
    int startNode = 0; // 从大图的节点0开始
    bool matchFound = mainGraph.bfsSubgraphMatching(subgraph, startNode);

    if (matchFound) {
        cout << "找到了匹配的子图" << endl;
    } else {
        cout << "未找到匹配的子图" << endl;
    }

    return 0;
}
