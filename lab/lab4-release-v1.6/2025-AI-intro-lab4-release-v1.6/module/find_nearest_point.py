import numpy as np

class TreeNode:
    def __init__(self, parent_idx, x, y):
        self.parent_idx = parent_idx
        self.pos = np.array([x, y])
        

def find_nearest_point(point, graph):
    """
    找到图中离目标位置最近的节点，返回该节点的编号和到目标位置距离、
    输入：
    point：维度为(2,)的np.array, 目标位置
    graph: List[TreeNode]节点列表
    输出：
    nearest_idx, nearest_distance 离目标位置距离最近节点的编号和距离
    """
    nearest_idx = -1
    nearest_distance = 10000000.
    ### 你的代码 ###
    graph_poss = []
    for node in graph:
        graph_poss.append(node.pos)
    arr_graph_poss  = np.array(graph_poss)
    len_graph = len(graph)
    arr_target = np.tile(point,(len_graph,1))
    arr_dis = np.linalg.norm(arr_graph_poss - arr_target, ord=2,axis=1)
    # 找出最小距离的下标
    nearest_idx = np.argmin(arr_dis)
    # 找到最近的点
    nearest_distance = arr_dis[nearest_idx]
    ### 你的代码 ###
    return nearest_idx, nearest_distance
# =====================
# 单元测试代码
# =====================
if __name__ == "__main__":
    graph = [
        TreeNode(-1, 0, 0),
        TreeNode(0, 1, 1),
        TreeNode(1, 2, 2),
        TreeNode(2, 5, 5),
        TreeNode(2, 1.1, 1.2),
    ]

    target_point = np.array([1.1, 1.2])
    idx, dist = find_nearest_point(target_point, graph)

    print("最接近的点编号:", idx)
    print("距离为:", dist)
    print("该点坐标为:", graph[idx].pos)