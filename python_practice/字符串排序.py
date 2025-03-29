#得写排序算法
num = float(input())
i=0
my_list = []
while i<num:
    i+=1
    text = input()
    flag = 0
    for item in my_list:
        if item>text:
            my_list.insert(my_list.index(item),text)
            flag = 1
            break
    if flag == 0:
        my_list.append(text)
for item in my_list:
    print(item)
                
            
            
