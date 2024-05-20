f = open("input.txt", "r")
dr = open("output.txt", "w")

ss = f.readlines()

for s in ss:
    s = s.rstrip('\r\n')
    ret = s[0:7]
    pos = s.find('\t', 1 + s.find('\t', 1 + s.find('\t', 1 + s.find('\t'))))
    ret += (74 - len(s) + pos) * '.' + s[pos+1:]
    dr.write(ret + '\n')

f.close()
dr.close()
