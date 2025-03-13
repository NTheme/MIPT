f = open("input.txt", "r")
dr = open("output.txt", "w")

ss = f.readlines()

d = []

for s in ss:
    s = s[0:len(s)-1] + "         d"
    for i in range(len(s) - 7):
        if s[i:i+5] == ".hlm " or s[i:i+6] == ".hlm. ":
            j = i
            while s[j] != ' ':
                j -= 1
            d.append(s[j+1:i+4])
        if s[i:i+6] == ".brhl " or s[i:i+7] == ".brhl. ":
            j = i
            while s[j] != ' ':
                j -= 1
            d.append(s[j+1:i+5])

for s in d:
  a = True
  for c in s:
    if c.isalpha():
      if not c.islower():
        a = False
  if s[0] != '.' and a:
    dr.write(s + '\n')

f.close()
dr.close()