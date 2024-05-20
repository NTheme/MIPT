f = open("input.txt", "r")
d = open("output.txt", "w")

s = f.readline()

nlower = 0
ncase = 0
ndigit = 0

for c in s:
  if c.isdigit():
    ndigit +=1
  elif c.islower():
    nlower +=1
  elif c.isupper():
    ncase += 1

if nlower > 0 and ncase > 0 and ndigit > 0:
  d.write("YES\n")
else:
  d.write("NO\n")
  
if ndigit <= 3 and len(s) <= 10:
  d.write("YES")
else:
  d.write("NO")
  
f.close()
d.close()