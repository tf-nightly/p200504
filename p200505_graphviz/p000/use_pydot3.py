import pydot

graph = pydot.Dot(graph_type="digraph")
def n(label):
  """Defines a new node."""
  node = pydot.Node(label)
  graph.add_node(node)
  return node
def e(src, dst):
  """Defines a new edge."""
  edge = pydot.Edge(src, dst)
  graph.add_edge(edge)
def ns(labels):
  """Defines new nodes."""
  nodes = []
  for label in labels:
    node = n(label)
    nodes.append(node)
  return nodes
def es(srcs, dsts):
  """Defines new edges."""
  for src in srcs:
    for dst in dsts:
      e(src, dst)

x, w, mul_x_w, b, add_mul_b, activation, y = ns("x w * b + ReLU y".split())
es([x, w], [mul_x_w])
es([b, mul_x_w], [add_mul_b])
e(add_mul_b, activation)
e(activation, y)

graph.write_png("png.png")




