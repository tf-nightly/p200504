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

W, V, B = ns("W V B".split())
phi_ = n("Y-1")
for i in range(3):
  a = "X{1} ・ ・ + Y{1}=phi()"
  a = a.format(i - 1, i)
  X, XW, Y_V, add, phi = ns(a.split())
  es([X, W], [XW])
  es([phi_, V], [Y_V])
  es([XW, Y_V, B], [add])
  e(add, phi)
  phi_ = phi

graph.write_png("png.png")




