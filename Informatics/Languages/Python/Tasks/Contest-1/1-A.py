import math

a = int(input())
b = int(input())

if a < 0 and b > 0:
    print(-(abs(a) % b))
else:
    print(a % b)
