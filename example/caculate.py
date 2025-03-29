import math
E = 1.6e11        # 杨氏模量（单位：Pa）
p = 2.33e3        # 密度（单位：kg/m³）

L = float(input())              # 长度（单位：m）#4e-4 #5e-6
W = float(input())              # 宽度（单位：m）#5e-6  #5e-7 
H = float(input())         # 高度（单位：m）#2e-6  #1e-8

k = (E * W * H**3) / (4 * L**3)  # 刚度计算
print(k)
m = (33/140)*p * W * H * L# 质量计算
print(m)
f = (1 / (2 * math.pi) )* (k / m)**0.5  # 共振频率公式
print(f)
mm = m +1e-18
f1 = (1 / (2 * math.pi)) * (k / mm)**0.5
print(f1)
_f1 = f1 - f
print("_f1",_f1)
mmm = m +2.5e-19
f2 = (1 / (2 * math.pi)) * (k / mmm)**0.5
print("f2",f2)
_f2 = f2 - f
print("_f2",_f2)

  # 输出结果（单位：Hz）