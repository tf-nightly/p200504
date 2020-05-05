"""
$$$
A = \left(
\begin{array}{ccc}
a & b & c \\
d & e & f \\
g & h & i
\end{array}
\right)
$$$
"""
H = 7
W = 6
s = ""
s += r"""$$$
W = \left(
\begin{array}{"""
s += W * "c"
s += """}
"""
for h in range(H):
  for w in range(W):
    s += "w_{{{}{}}}".format(h, w)
    if w != W - 1:
      s += " & "
  if h != H - 1:
    s += r" \\"
  s += "\n"
s += r"""\end{array}
\right)
$$$
"""

with open("p109_matjax.txt", "w") as f:
  f.write(s)

