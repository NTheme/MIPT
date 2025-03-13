n = int(input())
k = int(input())

arr = [list(set(str(input()).split())) for i in range(0, n)]
none = list(set(str(input()).split()))

dict = {}

for p in arr:
    for q in p:
        if q not in none:
            if q not in dict:
                dict[q] = 0
            dict[q] += 1


arr = sorted(dict.values())
for i in range(len(arr) - k, len(arr)):
    print(arr[i])
