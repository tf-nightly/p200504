import pydot

graph = pydot.Dot(graph_type="digraph")
def n(label):
  "Defines a new node."
  node = pydot.Node(label)
  graph.add_node(node)
  return node
def e(src, dst):
  "Defines a new edge."
  edge = pydot.Edge(src, dst)
  graph.add_edge(edge)

xs = []
for i in range(8):
  xs.append(n("x{}".format(i)))

ys = []
for i in range(11):
  ys.append(n("y{}".format(i)))

for x in xs:
  for y in ys:
    e(x, y)

graph.write_png("png.png")




