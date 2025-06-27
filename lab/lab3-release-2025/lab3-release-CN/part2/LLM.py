import re
from Maze import Maze
from openai import OpenAI


# TODO: Replace this with your own prompt.
your_prompt = """
你是一个专业的吃豆人游戏AI控制器，需要根据当前游戏状态选择最佳移动方向。请严格遵守以下规则：

1. 游戏目标：
- 尽可能多吃豆子，吃完后获胜
- 避开鬼魂


2. 移动规则：
- 只能选择当前可用的方向（由系统提供）
- 不能穿过墙壁（撞墙会死）
-当前情况鬼不会动（优先吃豆子）
-被鬼抓到会输掉比赛
- 如果被鬼魂追逐，优先逃跑

3. 决策策略：
- 优先选择有豆子的方向
- 如果多个方向有豆子，选择离鬼魂最远的方向
- 如果没有豆子，选择未走过的区域
- 被鬼魂接近时选择相反方向（若鬼移动）
- 探索未走过的区域

4. 位置关系：
- 当前吃豆人位置：{pacman_pos}
- 鬼魂位置：{ghost_pos}
- 可用方向：{available_directions}

5.得知条件
-地图布局
-吃豆人的位置
-鬼魂位置(可以比较前后鬼的位置判断鬼是否移动)
-曾经走过的位置
-可用的方向

6得分规则
-尽量在50步内吃完豆子，这样可以得到100分（满分100）
-高于50步，但低于90步，得分为60分
-碰到墙壁，高于90步，被鬼抓到为0分
"""

# Don't change this part.
output_format = """
输出必须是0-3的整数，上=0，下=1，左=2，右=3。
*重点*：(5,5)的上方是(4,5)，下方是(6,5)，左方是(5,4)，右方是(5,6)。
输出格式为：
“分析：XXXX。
动作：0（一个数字，不能出现其他数字）。”
"""

prompt = your_prompt + output_format


def get_game_state(maze: Maze, places: list, available: list) -> str:
    """
    Convert game state to natural language description.
    """
    #将游戏状态转换为自然语言描述
    description = ""
    for i in range(maze.height):
        for j in range(maze.width):
            description += f"({i},{j})="
            if maze.grid[i, j] == 0:
                description += f"空地"
            elif maze.grid[i, j] == 1:
                description += "墙壁"
            else:
                description += "豆子"
            description += ","
        description += "\n"
    places_str = ','.join(map(str, places))
    available_str = ','.join(map(str, available))
    state = f"""当前游戏状态（坐标均以0开始）：\n1、迷宫布局（0=空地,1=墙,2=豆子）：\n{description}\n2、吃豆人位置：{maze.pacman_pos[4]}\n3、鬼魂位置：{maze.pacman_pos[3]}\n4、曾经走过的位置：{places_str}\n5、可用方向：{available_str}\n"""
    return state


def get_ai_move(client: OpenAI, model_name: str, maze: Maze, file, places: list, available: list) -> int:
    """
    Get the move from the AI model.
    :param client: OpenAI client instance.
    :param model_name: Name of the AI model.
    :param maze: The maze object.
    :param file: The log file to write the output.
    :param places: The list of previous positions.
    :param available: The list of available directions.
    :return: The direction chosen by the AI.
    """
    state = get_game_state(maze, places, available)

    file.write("________________________________________________________\n")
    file.write(f"message:\n{state}\n")
    print("________________________________________________________")
    print(f"message:\n{state}")

    print("Waiting for AI response...")
    all_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": state
            }
        ],#列表里面有两个字典
        stream=False,
        temperature=.0
    )
    info = all_response.choices[0].message.content#API 返回的响应对象中的 choices 字段，是一个列表，包含模型生成的所有候选响应
    #choices[0]: 第一个候选响应。
    #message: 响应中的消息对象
    #content: 消息的文本内容
    file.write(f"AI response:\n{info}\n")
    print(f"AI response:\n{info}")
    numbers = re.findall(r'\d+', info) # 提取所有数字
    choice = numbers[-1] # 取最后一个数字（假设是方向）
    return int(choice), info
