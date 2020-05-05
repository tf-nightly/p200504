import pydot

graph = pydot.Dot(graph_type="digraph")
id = 0
def n(label):
  """Defines a new node."""
  global id
  node = pydot.Node(name=id, obj_dict=None, label=label)
  id += 1
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

x0, x1, x2, w0, w1, w2, m0, m1, m2, b, add, phi = ns("x0 x1 x2 w0 w1 w2 ・ ・ ・ b + phi".split())
es([x0, w0], [m0])
es([x1, w1], [m1])
es([x2, w2], [m2])
es([m0, m1, m2, b], [add])
e(add, phi)

graph.write_png("png.png")




