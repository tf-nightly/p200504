import pydot

graph = pydot.Dot(graph_type="digraph")

xs = []
for i in range(3):
  label = "x{}".format(i)
  x = pydot.Node(label)
  graph.add_node(x)
  xs.append(x)

ys = []
for i in range(4):
  label = "y{}".format(i)
  y = pydot.Node(label)
  graph.add_node(y)
  ys.append(y)

for x in xs:
  for y in ys:
    edge = pydot.Edge(x, y)
    graph.add_edge(edge)

graph.write_pdf("pdf.pdf")




