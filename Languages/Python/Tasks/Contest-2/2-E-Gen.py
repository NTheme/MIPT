file = open('gen.txt', 'w')

file.write('[')

i = 1
cnt = 0
while cnt < 10000:
  cnt3 = 0
  icopy = i
  while icopy > 0:
    if icopy % 10 == 3:
      cnt3 += 1
    icopy //= 10
  if cnt3 == 3:
    file.write(f'{i}')
    if cnt != 9999:
      file.write(", ")
    else:
      file.write("]")
    cnt += 1
  i += 1

file.close()
