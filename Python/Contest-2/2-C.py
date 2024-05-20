l = int(input())
str = str(input())

if len(str) < l:
    print(str)
else:
    print(str[0:l], end='')
    print()

    for i in range(l, len(str)):
        print("&" * (l - 1), end='')
        print(str[i])
