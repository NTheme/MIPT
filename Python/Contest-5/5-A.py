s = str(input())

res = ''
for i, c in enumerate(s):
    if i % 2 == 1:
        res += c.upper()
    else:
        res += c.lower()

print(res)
