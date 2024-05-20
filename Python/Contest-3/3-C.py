s = str(input())
n = int(input())

dict = {}
for i in range(0, len(s) - n + 1):
    dict[s[i:i+n]] = s.count(s[i:i+n])

ans = []
for c in dict:
    if dict[c] > 1:
        ans.append(c)

print(sorted(ans))
