n = 4
for i in range(2, 51):
  s = "{}桁の{}進数は（10進数で）{:,}までの数を表せます。"
  s = s.format(n, i, i**n - 1)
  print(s)
