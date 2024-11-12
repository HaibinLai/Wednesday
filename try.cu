#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

using namespace std;

// 图类，使用邻接表表示
class Graph {
public:
    vector<vector<int>> adjList;

    // 构造函数，初始化图的节点数量
    Graph(int nodes) {
        adjList.resize(nodes);
    }

    // 添加边
    void addEdge(int u, int v) {
        // 如果边已经存在，则不重复添加
        if (find(adjList[u].begin(), adjList[u].end(), v) == adjList[u].end()) {
            adjList[u].push_back(v);
        }
        if (find(adjList[v].begin(), adjList[v].end(), u) == adjList[v].end()) {
            adjList[v].push_back(u);
        }
    }

    // 删除边
    void removeEdge(int u, int v) {
        // 从u的邻接表中删除v
        adjList[u].erase(remove(adjList[u].begin(), adjList[u].end(), v), adjList[u].end());
        // 从v的邻接表中删除u
        adjList[v].erase(remove(adjList[v].begin(), adjList[v].end(), u), adjList[v].end());
    }

    // 更新边：删除旧的边，添加新的边
    void updateEdge(int old_u, int old_v, int new_u, int new_v) {
        // 先删除旧的边
        removeEdge(old_u, old_v);
        // 添加新的边
        addEdge(new_u, new_v);
    }

    // 打印图的邻接表
    void printGraph() {
        for (int i = 0; i < adjList.size(); i++) {
            cout << "Node " << i << ": ";
            for (int neighbor : adjList[i]) {
                cout << neighbor << " ";
            }
            cout << endl;
        }
    }
};

// 子图匹配算法
bool isSubgraphMatch(const Graph& mainGraph, const Graph& subGraph, 
                     unordered_map<int, int>& mapping, const unordered_set<int>& subGraphNodes, int subNode) {
    // 检查当前子图节点的所有邻居是否在主图中对应匹配
    for (int neighbor : subGraph.adjList[subNode]) {
        bool matchFound = false;
        for (auto& entry : mapping) {
            if (entry.second == neighbor) {
                matchFound = true;
                break;
            }
        }
        if (!matchFound) return false;
    }
    return true;
}

// 回溯搜索函数，寻找子图在主图中的匹配
bool subgraphDFS(const Graph& mainGraph, const Graph& subGraph, 
                 unordered_map<int, int>& mapping, const unordered_set<int>& subGraphNodes,
                 int subNode) {
    if (subNode == subGraphNodes.size()) {
        return true;  // 如果所有子图节点已经匹配，返回成功
    }

    for (int node : subGraphNodes) {
        if (mapping.find(node) == mapping.end()) {  // 如果该子图节点还未被映射
            // 尝试将子图的node映射到主图的某个节点
            for (int mainNode = 0; mainNode < mainGraph.adjList.size(); ++mainNode) {
                mapping[node] = mainNode;
                if (isSubgraphMatch(mainGraph, subGraph, mapping, subGraphNodes, node)) {
                    if (subgraphDFS(mainGraph, subGraph, mapping, subGraphNodes, subNode + 1)) {
                        return true;
                    }
                }
                // 回溯
                mapping.erase(node);
            }
        }
    }
    return false;
}

int main() {
    // 创建一个包含6个节点的图
    Graph mainGraph(6);

    // 添加一些边
    mainGraph.addEdge(4, 5);
    mainGraph.addEdge(0, 1);
    mainGraph.addEdge(0, 2);
    mainGraph.addEdge(1, 3);
    mainGraph.addEdge(2, 3);
    
    mainGraph.addEdge(3, 4);
    // mainGraph.addEdge(4, 5);

    cout << "Initial Graph:" << endl;
    mainGraph.printGraph();

    // 更新边(0, 1) 到 (0, 3)
    cout << "\nUpdating edge (0, 1) to (0, 3)..." << endl;
    mainGraph.updateEdge(0, 1, 0, 3);

    cout << "\nGraph after edge update:" << endl;
    mainGraph.printGraph();

    // 删除边 (2, 3)
    cout << "\nRemoving edge (2, 3)..." << endl;
    mainGraph.removeEdge(2, 3);

    cout << "\nGraph after edge removal:" << endl;
    mainGraph.printGraph();

     // 子图构造
    Graph subGraph(3);  // 创建一个包含3个节点的子图
    subGraph.addEdge(0, 1);
    subGraph.addEdge(1, 2);

    cout << "Sub Graph:" << endl;
    subGraph.printGraph();
    
    // 子图节点集合
    unordered_set<int> subGraphNodes = {0, 1, 2};  // 子图节点

    // 用于记录匹配的映射
    unordered_map<int, int> mapping;
    
    // 尝试进行子图匹配
    if (subgraphDFS(mainGraph, subGraph, mapping, subGraphNodes, 0)) {
        cout << "Subgraph matched!" << endl;
        for (auto& entry : mapping) {
            cout << "Subgraph node " << entry.first << " is matched with main graph node " << entry.second << endl;
        }
    } else {
        cout << "No subgraph match found." << endl;
    }

    return 0;
}
