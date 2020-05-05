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

x = n("x")
w = n("w")
mul_x_w = n("mul")
e(x, mul_x_w)
e(w, mul_x_w)
b = n("b")
add_mul_b = n("add")
e(b, add_mul_b)
e(mul_x_w, add_mul_b)
activation = n("ReLU")
e(add_mul_b, activation)
y = n("y")
e(activation, y)


graph.write_png("png.png")




