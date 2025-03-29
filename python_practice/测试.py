
a=eval(input("请输入一个大于6的偶数："))
if a<6:
    print("请输入一个大于6的偶数：")
else:
    while a%2!=0:
        print("请输入一个大于6的偶数：")
        break
    else:
        for i in range(3,int((a-2)/2)):
            for j in range(2,i):
                if i%j==0:
                    break
            else:
                z=a-i
                for m in range(2,z):
                    if z%m==0:
                         break
                else:
                    print(f"{a}={i}+{z}")

