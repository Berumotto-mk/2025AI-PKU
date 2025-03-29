#首先把两个字符串读取出来，然后比较两个字符串长度
text = input()
parts = text.split()
firststring = parts[0]
secondstring = parts[1]
if len(firststring) >= len(secondstring):
    result =firststring + secondstring
else:
    result = secondstring + firststring
print(result)
