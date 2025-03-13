xa = int(input())
ya = int(input())
xb = int(input())
yb = int(input())
xc = int(input())
yc = int(input())

if min(xa, xb) <= xc and xc <= max(xa, xb) and min(ya, yb) <= yc and yc <= max(ya, yb):
  print("False")
else:
  print(f'{2*xc - xa}\n{2*yc - ya}\n{2*xc - xb}\n{2*yc - yb}')
