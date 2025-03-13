import sys

print("? -1000000 1000000")
sys.stdout.flush()
x1 = int(input())
sys.stdout.flush()

print("? 1000000 1000000")
sys.stdout.flush()
x2 = int(input())
sys.stdout.flush()

y = int(2000000 - (x1 + x2) / 2)
x = int((x1 - x2) / 2)
print(f'! {x} {y}')
sys.stdout.flush()
