import numpy as np
from typing import List
from utils import TreeNode
from simuScene import PlanningMap


### 定义一些你需要的变量和函数 ###
STEP_DISTANCE = 3
TARGET_THREHOLD = 0.23#target_threhold
ONE_TARGET_MOVE_NUM = 30
def edge(walls):
    x_min=np.min(walls,axis=0)[0]
    y_min=np.min(walls,axis=0)[1]
    x_max=np.max(walls,axis=0)[0]
    y_max=np.max(walls,axis=0)[1]
    return (x_min,x_max,y_min,y_max)
def distance(current_position,target):
    '''计算两点之间的距离'''
    return (current_position-target)@(current_position-target).T
### 定义一些你需要的变量和函数 ###


class RRT:
    def __init__(self, walls) -> None:
        """
        输入包括地图信息，你需要按顺序吃掉的一列事物位置 
        注意：只有按顺序吃掉上一个食物之后才能吃下一个食物，在吃掉上一个食物之前Pacman经过之后的食物也不会被吃掉
        """
        self.map = PlanningMap(walls)
        self.walls = walls
        
        # 其他需要的变量
        ### 你的代码 ###      
        self.cur_target_index=1
        self.cur_target_moved_num=0
        self.my_next_food=np.array([0,0])
        self.lastpos=None
        ### 你的代码 ###
        
        # 如有必要，此行可删除
        self.path = None
        
        
    def find_path(self, current_position, next_food):
        """
        在程序初始化时，以及每当 pacman 吃到一个食物时，主程序会调用此函数
        current_position: pacman 当前的仿真位置
        next_food: 下一个食物的位置
        
        本函数的默认实现是调用 build_tree，并记录生成的 path 信息。你可以在此函数增加其他需要的功能
        """
        
        ### 你的代码 ###      
        self.cur_target_index=1
        self.cur_target_moved_num=0
        ### 你的代码 ###
        # 如有必要，此行可删除
        self.path = self.build_tree(current_position, next_food)
        
        
    def get_target(self, current_position, current_velocity):
        """
        主程序将在每个仿真步内调用此函数，并用返回的位置计算 PD 控制力来驱动 pacman 移动
        current_position: pacman 当前的仿真位置
        current_velocity: pacman 当前的仿真速度
        一种可能的实现策略是，仅作为参考：
        （1）记录该函数的调用次数#怎样记录次数
        （2）假设当前 path 中每个节点需要作为目标 n 次#怎样确定其执行的次数
        （3）记录目前已经执行到当前 path 的第几个节点，以及它的执行次数，如果超过 n，则将目标改为下一节点
        
        你也可以考虑根据当前位置与 path 中节点位置的距离来决定如何选择 target
        
        同时需要注意，仿真中的 pacman 并不能准确到达 path 中的节点。你可能需要考虑在什么情况下重新规划 path
        """
        target_pose = np.zeros_like(current_position)
        ### 你的代码 ###
        '''每次调用的 target 通常是 RRT 中的一个节点，但是如果在 build tree之后优化，path，只需要考虑 path 中的某一个节点'''
        cur_target=self.path[self.cur_target_index]
        if(not self.map.checkline(current_position.tolist(),self.path[-1].tolist())[0]):
            #print("In a empty straight so rush to it")
            return self.path[-1]
        if distance(current_position,self.path[-1])<0.1:
            return self.path[-1]
        if self.cur_target_moved_num<ONE_TARGET_MOVE_NUM:
            target_pose=cur_target
            self.cur_target_moved_num+=1
        else:
            #到达一个短期的 target 之后更换目标；
            '''新发现了一个 bug 就是会被堵在墙角，然后 规划的路径显然无法通过'''
            if  current_velocity[0]<1e-3 and current_velocity[1]<1e-3 and ( self.map.checkline(current_position.tolist(),cur_target.tolist())[0]):
                '''因为特殊原因导致进行过程中无法抵达之前规划的路径，重新规划路径'''
                #print("because of the wall refind path")
                self.find_path(current_position,self.path[-1])
                mid_change_target=self.path[self.cur_target_index]
                self.cur_target_moved_num+=1
                return mid_change_target
            '''更换目标'''
            self.cur_target_index+=1
            '''清零计次'''
            self.cur_target_moved_num=0
            if(self.cur_target_index>=len(self.path)):
                '''如果 out of range,重新规划路径'''
                #print("out of range")
                self.find_path(current_position,self.path[-1])
                mid_change_target=self.path[self.cur_target_index]
                self.cur_target_moved_num+=1
                return mid_change_target
            #print(self.cur_target_index,"move_time",self.cur_target_moved_num)
            target_pose=self.path[self.cur_target_index]
        ### 你的代码 ###
        return target_pose-0.1*current_velocity
        
    ### 以下是RRT中一些可能用到的函数框架，全部可以修改，当然你也可以自己实现 ###
    def build_tree(self, start, goal):
        """
        实现你的快速探索搜索树，输入为当前目标食物的编号，规划从 start 位置食物到 goal 位置的路径
        返回一个包含坐标的列表，为这条路径上的pd targets
        你可以调用find_nearest_point和connect_a_to_b两个函数
        另外self.map的checkoccupy和checkline也可能会需要，可以参考simuScene.py中的PlanningMap类查看它们的用法
        """
        path = []
        graph: List[TreeNode] = []
        graph.append(TreeNode(-1, start[0], start[1]))
        ### 你的代码 ###
        now_node = graph[0]
        '''构建RRT'''
        def get_rand():
            cur_edge=edge(self.walls)
            if np.random.randint(0,100)>=80:
                rand=goal#goal也是np.array
            else:
                x = np.random.uniform(cur_edge[0], cur_edge[1])
                y = np.random.uniform(cur_edge[2], cur_edge[3])
                rand=np.array([x,y])
            return rand
        
        while distance(goal,now_node.pos) > TARGET_THREHOLD:
            random_point =get_rand()
            nearest_idx,nearest_dis = self.find_nearest_point(random_point,graph)
            if np.allclose(graph[nearest_idx].pos, random_point):
                continue
            if nearest_dis <= STEP_DISTANCE:
                '''如果距离很小，就考虑将 rand 节点加入到树上'''
                if not self.map.checkline(graph[nearest_idx].pos.tolist(),random_point.tolist())[0] :
                    graph.append(TreeNode(nearest_idx,random_point[0],random_point[1]))
                    
            else:
                '''如果相距大于步长，前进 STEP_DISTANCE'''
                isempty,next_point = self.connect_a_to_b(graph[nearest_idx].pos,random_point)
                if isempty:
                    graph.append(TreeNode(nearest_idx,next_point[0],next_point[1]))
            now_node = graph[-1]
        
        
            
        '''以上构建好了一棵树，现在利用回溯的手段将树上必要的节点加入到 ori_path 中'''
        ori_path = []
        if not np.array_equal(graph[-1].pos,goal):
            ori_path.append(goal)
        now_node = graph[-1]
        root = now_node.parent_idx
        ori_path.append(now_node.pos)
        while root != -1:
            ori_path.append(graph[root].pos)
            root = graph[root].parent_idx
        ori_path = list(reversed(ori_path))
        
        '''优化ori_path为path'''
        '''逻辑为'''
        path.append(start)
        cur_idx = 0
        ori_len = len(ori_path)
        while not np.array_equal(path[-1],goal):
            for i in range(ori_len - 1,cur_idx,-1):
                if not self.map.checkline(ori_path[cur_idx].tolist(),ori_path[i].tolist())[0]:
                    path.append(ori_path[i])
                    cur_idx = i
                    break        
        ### 你的代码 ###
        
        return path

    @staticmethod
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
        graph_len=len(graph)
        for i in range(graph_len):
            dis=distance(point,graph[i].pos)
            if dis<nearest_distance:
                nearest_distance=dis
                nearest_idx=i
        ### 你的代码 ###
        return nearest_idx, nearest_distance
    
    def connect_a_to_b(self, point_a, point_b):
        """
        以A点为起点，沿着A到B的方向前进STEP_DISTANCE的距离，并用self.map.checkline函数检查这段路程是否可以通过
        输入：
        point_a, point_b: 维度为(2,)的np.array，A点和B点位置，注意是从A向B方向前进
        输出：
        is_empty: bool，True表示从A出发前进STEP_DISTANCE这段距离上没有障碍物
        newpoint: 从A点出发向B点方向前进STEP_DISTANCE距离后的新位置，如果is_empty为真，之后的代码需要把这个新位置添加到图中
        """#没有说明新位置是np.array还是list
        is_empty = False
        newpoint = np.zeros(2)
        ### 你的代码 ###
        newpoint = (STEP_DISTANCE*(point_b-point_a))/distance(point_a,point_b)+point_a
        if not self.map.checkline(point_a.tolist(),newpoint.tolist())[0] and not self.map.checkoccupy(newpoint):
            is_empty = True
        else:
            is_empty = False
        ### 你的代码 ###
        return is_empty, newpoint
