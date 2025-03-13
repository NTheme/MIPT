n = int(input())
m = int(input())
s = list()

c = str()
while c != "0":
    c = str(input())
    s.append(c)
s.pop()

word = ""
cnt = 0
for r in s:
    if len(word) + len(r) > n or cnt >= m:
        print(word[0:len(word)-1])
        word = ""
        cnt = 0
    word += r + ' '
    cnt += 1

print(word[0:len(word)-1])
