a = int(input())
b = int(input())

res = True

if a == 0:
    res = False

while a > 0:
    if a % 2 == 1 and b % 2 == 1:
        res = False
        break
    a = a // 2
    b = b // 2

print(res)
