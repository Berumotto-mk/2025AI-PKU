import math
def entropy(kind_list):
    result = 0
    sum = 0
    for item in kind_list:
        sum+=item
    for item in  kind_list:
        result += -item/sum * math.log(item/sum,2)
    return result
A1_list = [2,1]
A2_list = [2,4]
output1 = entropy(A1_list)
print("A1_list的熵为：",output1)
output2 = entropy(A2_list)
print("A2_list的熵为：",output2)
gain = 0.99 - 3/9 * output1 - 6/9 * output2
print("信息增益为：",gain)
entropy_A = entropy([3,6])
print("A的熵为：",entropy_A)
gain_ratio = gain / entropy_A
print("增益率为：",gain_ratio)
