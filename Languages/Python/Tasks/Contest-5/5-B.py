n = int(input())
m = int(input())
s = ["*   *",
     "*  **",
     "* * *",
     "**  *",
     "*   *"]

for i in range(n):
    for j in range(5):
        print((s[j] + " ") * (m - 1) + s[j])
    if i < n - 1:
        print(" " * len((s[0] + " ") * (m - 1) + s[0]))
