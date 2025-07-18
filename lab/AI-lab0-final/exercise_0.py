import numpy as np

balabala = 'balabala'

def b1(a):
    '''    
    
    输入： 一个整数list，计算其中偶数下标元素的和
    输出： 一个整数
    '''
    ans = 0
    for i in range(len(a)):
        if i%2 == 0:
            ans += a[i]
    return ans

def b2(scores:dict):
    '''
    输入：一个dict，形如： { "Alice": 100, "Bob": 10 }，包含人名和对应的成绩
    输出：一个list，包括所有成绩大于50的人，并按python字符串的默认顺序，从小到大对名字排序
    '''
    
    ret = None
    filtered_names = [name for name, score in scores.items() if score > 50]
    
    # 按名字排序
    filtered_names.sort()
    ret = filtered_names
    return ret        

def q1(shape):
    '''
    输入： shape, 一个整数tuple，表示数组的形状
    输出： 一个全为0的numpy数组，形状为shape，数据类型为np.float32
    '''
    x = balabala
    zero_array = np.zeros(shape, dtype=np.float32)
    x = zero_array
    #print(x)
    return x

def q2(n):
    '''
    输入： n，一个整数
    输出： 形状为(n,n)的矩阵，第i行第j列的元素值为i+j
    注意： i和j的取值范围都是[0,n)，请避免使用for循环(理论上并不需要任何for循环)
    '''
    #广播机制的使用
    x = balabala
    i = np.arange(n).reshape(n, 1)  # 创建一个列向量 [0, 1, 2, ..., n-1]
    j = np.arange(n)  # 创建一个行向量 [0, 1, 2, ..., n-1]
    matrix = i + j  # 利用广播机制生成矩阵
    x = matrix
    #print(x)
    return x

def q3(name_score_dict):
    '''
    输入： name_score_dict，一个字典，key是学生姓名，value是他们的分数
    示例：
    {
        "Alice": 90,
        "Bob": 80,
        "Cindy": 70,
    }
    输出： 定义一个Student类，包含两个属性：name和score，分别表示学生的姓名和分数。 
          为每个学生创建一个Student对象，并将这些对象放入一个列表中，按成绩从高到低排序。然后返回该列表
    '''
    
    res = []
    from collections import OrderedDict
    ordered_dict = OrderedDict(name_score_dict)
    name_score_dict = dict(sorted(ordered_dict.items(), key=lambda x: x[1], reverse=True))
    class Student:
         def __init__(self,name,score):
             self.name = name
             self.score = score
    for name,score in name_score_dict.items():
        res.append(Student(name,score))
    return res

def q4(a,b):
    '''
    输入： a, b, 两个形状相同的numpy数组
    输出： 将a中小于b的元素替换为b中对应的元素，然后返回替换后的数组
    注意： a[a<b] = b[a<b]这种操作当然也能实现，但是副作用是会改变原数组的值，可以参考np.where
    '''
    res = balabala
    res = np.where(a < b, b, a)
    return res

def q5():
    '''
    随机采样估计圆周率, 误差小于0.01即可
    输入： 无
    输出： 一个浮点数，表示圆周率
    注意：可以使用np.random.uniform产生随机数
    可使用方法比如：正方形内随机采样点，计算在圆内的点的比例
    '''
    
    res = balabala
    num_samples = 1000000
    
    # 在 [-1, 1] 范围内生成 num_samples 个随机点
    x = np.random.uniform(-1, 1, num_samples)
    y = np.random.uniform(-1, 1, num_samples)
    
    # 计算每个点到圆心 (0, 0) 的距离
    distances = np.sqrt(x**2 + y**2)
    
    # 判断点是否在圆内
    points_inside_circle = distances <= 1
    
    # 计算圆内点的数量
    num_points_inside_circle = np.sum(points_inside_circle)
    
    # 估计圆周率
    pi_estimate = (num_points_inside_circle / num_samples) * 4
    
    res = pi_estimate
    return res


class Layout:
    def __init__(self, layoutText=None) -> None:
        """
        初始化Layout对象。
        :param layoutText: 一个字符串列表，每个字符串代表地图的一行。
        """
        self.height = 0  # 地图的高度
        self.width = 0   # 地图的宽度
        self.walls = []  # 墙壁的位置列表，每个位置是一个(x, y)元组
        self.pacman_pos = None  # Pac-Man的位置，格式为(x, y)
        self.foods = []  # 食物的位置列表，每个位置是一个[x, y]的NumPy数组
        if layoutText is not None:
            self.load_layout(layoutText)  # 如果提供了布局文本，就加载布局

    def load_layout(self, layoutText) -> None:
        """
        从文本加载布局信息，更新地图的尺寸、墙壁、食物和Pac-Man的位置。
        :param layoutText: 一个字符串列表，每个字符串代表地图的一行。
        """
        self.height = len(layoutText)  # 地图高度为行数
        self.width = 0
        for y in range(self.height):
            self.width = max(self.width, len(layoutText[y]))  # 寻找最宽的行作为地图宽度
            for x in range(len(layoutText[y])):
                self.processLayoutChar(y, x, layoutText[y][x])  # 处理每个字符

    def processLayoutChar(self, x, y, layoutChar):
        """
        根据布局字符更新墙壁、食物和Pac-Man的位置。
        :param x: 字符的横坐标
        :param y: 字符的纵坐标
        :param layoutChar: 布局中的字符 ('%', '.', 'P', ' ')
        """
        if layoutChar == '%':
            self.walls.append((x, y))  # 墙壁
        elif layoutChar == '.':
            self.foods.append([x, y])  # 食物
        elif layoutChar == 'P':
            self.pacman_pos = (x, y)  # Pac-Man位置
        elif layoutChar == ' ':
            pass  # 空白位置不做处理
        else:
            raise NotImplementedError  # 遇到未知字符抛出异常

    def get_empty_map(self):
        """
        创建并返回一个新的Layout实例，它包含与当前实例相同的尺寸和墙壁位置，但不包含食物和Pac-Man的位置。
        :return: 一个新的Layout实例。
        """
        new_map = Layout()
        new_map.height = self.height
        new_map.width = self.width
        new_map.walls = self.walls.copy()  # 使用墙壁位置的副本以防止修改原始数据
        return new_map
    
    @staticmethod#静态方法不需要实例化就可以调用
    def build_layout(file_path):
        """
        从文件中读取布局文本并创建一个Layout实例。
        :param file_path: 布局文件的路径。
        :return: 一个根据文件内容创建的Layout实例。
        """
        with open(file_path, 'r') as f:
            layoutText = f.read().split('\n')  # 读取文件并按行分割
        return Layout(layoutText)

    def recover_layoutText(self):
        """
        将当前布局信息转换回文本格式。这里需要你的实现。
        注意：此方法中可能需要考虑x, y坐标的转置问题。
        """
        # 你的实现代码
        res =[' ' * self.width for _ in range(self.height)]
        for x,y in self.walls:
            res[x] = res[x][:y] + '%' + res[x][y+1:]#python中字符串不可变，要拆分为3部分进行插入
        for x,y in self.foods:
            res[x] = res[x][:y] + '.' + res[x][y+1:]
        if self.pacman_pos is not None:
            x,y = self.pacman_pos#解包单个元组时不同于解包多个元组
            res[x] = res[x][:y] + 'P' + res[x][y+1:]
        for i in range(self.height):
            res[i] +='\n'
        return ''.join(res)#将列表转换为字符串

    
def q6(file_path):
    '''
    上面定义了一个Layout类，用于表示游戏地图。
    其会读取一个字符串，每个字符代表一个格子的状态，其中：
    '%'代表墙，'.'代表食物，'P'代表pacman的初始位置，' '代表空格
    令人沮丧的是，这个类输入LayoutText以后就将原始文本转化成了只存贮物体坐标值的紧凑格式
    而我们希望你能够实现一个recover_layoutText方法，用于从紧凑形式返回原始的layoutText
    输入： file_path，文本文件的路径
    紧凑格式： 见类内的wall, foods, pacman_pos属性 和 processLayoutChar方法
    输出： 一个Layout对象，代表地图,我们会调用recover_layoutText方法，检查其是否正确
    输出例子：
        %%%%%%%%%%%
        %    P  ..%
        %.%%%%%%% %
        %       ..%
        %%%%%%%%%%%
    注意：本函数输入是一个file_path，代表地图的文本文件，你需要使用build_layout方法构造一个Layout对象
    
    
    '''
    balabala = Layout.build_layout(file_path)
    
    return balabala
    