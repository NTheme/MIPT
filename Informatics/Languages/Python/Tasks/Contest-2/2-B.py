n = int(input())
m = int(input())

row = ["alpha", "beta", "gamma", "delta", "epsilon",
       "zeta", "eta", "theta", "iota", "kappa"]
column = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]

for r in range(0, n):
    for c in range(0, m):
        print(row[r] + "_" + column[9] * (c // 10) + column[c % 10], end=('' if c == m - 1 else ' '))
    print()
