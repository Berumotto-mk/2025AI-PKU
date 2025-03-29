# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import heapq# 优先队列

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).# 这个类概述了搜索问题的结构，但没有实现任何方法（在面向对象的术语中：一个抽象类）。

    You do not need to change anything in this class, ever.# 您永远不需要更改此类中的任何内容。
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        # 返回一个三元组列表，其中'successor'是当前状态的后继状态，'action'是到达那里所需的操作，
        # 'stepCost'是扩展到该后继状态的增量成本。
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take# 要执行的操作列表

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.# 此方法返回特定操作序列的总成本。序列必须由合法移动组成。
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    # 搜索树中最深的节点。
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    # 搜索算法需要返回到达目标的操作列表。确保实现图搜索算法。
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    # 要开始，您可能希望尝试一些简单的命令来了解传递的搜索问题：
    print("Start:", problem.getStartState())#
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))#
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    #需要写一个dfs的代码
    Frontier = util.Stack()
    Visited = []
    Frontier.push((problem.getStartState(),[]))
    while not Frontier.isEmpty():
        state,actions = Frontier.pop()
        Visited.append(state)#什么时候算作已经完全访问过了，从栈中弹出
        if problem.isGoalState(state):
            return actions
        for next in problem.getSuccessors(state):
            n_state = next[0]
            n_direction = next[1]
            if n_state not in Visited:
                Frontier.push((n_state,actions+[n_direction]))
    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #写一个bfs的代码
    # 构建一个队列
    Frontier = util.Queue()
    # 创建已访问节点集合
    Visited = []
    # 将(初始节点,空动作序列)入队
    Frontier.push( (problem.getStartState(), []) )#将一个元组入队，第一个元素是状态，第二个元素是动作序列
    # 将初始节点标记为已访问节点
    Visited.append( problem.getStartState() )#将初始节点加入到已经访问的节点集合中
    # 判断队列非空
    while Frontier.isEmpty() == 0:
        # 从队列中弹出一个状态和动作序列
        state, actions = Frontier.pop()#弹出一个元组，第一个元素是状态，第二个元素是动作序列
        # 判断是否为目标状态，若是则返回到达该状态的累计动作序列
        if problem.isGoalState(state):
            return actions 
        # 遍历所有后继状态
        for next in problem.getSuccessors(state):
            # 新的后继状态
            n_state = next[0]
            # 新的action
            n_direction = next[1]
            # 若该状态没有访问过
            if n_state not in Visited:
                # 计算到该状态的动作序列，入队
                Frontier.push( (n_state, actions + [n_direction]) )
                Visited.append( n_state )
    util.raiseNotDefined()
    
#! 例题答案如下
# def breadthFirstSearch(problem):
#     """Search the shallowest nodes in the search tree first."""
#     #python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs --frameTime 0
#     #python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5 --frameTime 0
#     "*** YOUR CODE HERE ***"

#     Frontier = util.Queue()
#     Visited = []
#     Frontier.push( (problem.getStartState(), []) )
#     #print 'Start',problem.getStartState()
#     Visited.append( problem.getStartState() )

#     while Frontier.isEmpty() == 0:
#         state, actions = Frontier.pop()
#         if problem.isGoalState(state):
#             #print 'Find Goal'
#             return actions 
#         for next in problem.getSuccessors(state):
#             n_state = next[0]
#             n_direction = next[1]
#             if n_state not in Visited:
                
#                 Frontier.push( (n_state, actions + [n_direction]) )
#                 Visited.append( n_state )

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # 写一个ucs的代码
    #和深搜不一样的地方，要考虑成本，
    #优先扩展最小成本的节点
    #一个stated的cost是从初始状态到该状态的cost,使用getCostOfActions(actions)函数
    #problem是一个SearchProblem对象
    Frontier = util.PriorityQueue()
    Frontier.push((problem.getStartState(),[]),0)
    expended = []
    while not Frontier.isEmpty():
        state,actions = Frontier.pop()
        if problem.isGoalState(state):
            return actions
        if state in expended:#如果已经扩展过了，就不再扩展,为什么会出次问题，
            #因为有可能会有多个路径到达同一个节点，但是不同的路径下访问到的点
            continue
        expended.append(state)
        for next in problem.getSuccessors(state):
            n_state = next[0]
            n_direction = next[1]
            n_cost = next[2]
            if n_state not in expended and n_state not in Frontier.heap:
                Frontier.push((n_state,actions+[n_direction]),problem.getCostOfActions(actions+[n_direction]))
            elif n_state in Frontier.heap:
                Frontier.update((n_state,actions+[n_direction]),problem.getCostOfActions(actions+[n_direction]))            
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    # 启发式函数估计从当前状态到提供的SearchProblem中最近目标的成本。这个启发式是微不足道的。
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # 写一个astar的代码
    #astar算法是在ucs算法的基础上加上了一个启发函数f(n) = g(n)+h(n)
    Frontier =util.PriorityQueueWithFunction\
        (lambda x:problem.getCostOfActions(x[1])+heuristic(x[0],problem))
    start = problem.getStartState()
    Frontier.push((start,[]))
    expended = []
    
    while not Frontier.isEmpty():
        state,actions = Frontier.pop()
        
        if problem.isGoalState(state):
            return actions
        if state in expended:#如果已经扩展过了，就不再扩展,为什么会出次问题，
            continue
        #因为有可能会有多个路径到达同一个节点，但是不同的路径下访问到的点
        expended.append(state)
        for next in problem.getSuccessors(state):
            n_state = next[0]
            n_direction = next[1]
            if n_state not in expended and n_state not in Frontier.heap:
                Frontier.push((n_state,actions+[n_direction]))
            elif n_state in Frontier.heap:
                Frontier.update((n_state,actions+[n_direction]),\
                    problem.getCostOfActions(actions+[n_direction])+heuristic(n_state,problem))
    util.raiseNotDefined()

# Abbreviations# 缩写
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
