s = str(input())

found = False
for c in s:
    if s.count(c) > 1:
        found = True
        break

print(found)
