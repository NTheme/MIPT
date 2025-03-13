f = open("index.h", "w")
f.write("#pragma once\n#include <iostream>\n\nint arr[10] = { ")

a = 0
b = 1
for i in range(10):
    f.write(f"{a}")
    if (i != 9):
        f.write(", ")
    a, b = b, a + b

f.write(" };\n")
f.close()

