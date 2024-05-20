n = int(input())

dp_a = [0, 1]
dp_n = [1, 2]

for i in range(2, n + 1):
    dpn = (dp_a[i - 1] + dp_n[i - 1]) * 2
    dpa = dp_n[i - 1]
    dp_n.append(dpn)
    dp_a.append(dpa)

print(dp_n[n] + dp_a[n])
