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

X, Y_ = ("X(t) Y(t-1)".split())

A0, W0, XW0, V0, Y_V0, B0, add0 = ns("A0=sigmoid() W0 ・ V0 ・ B0 +".split())
es([X, W0], [XW0])
es([Y_, V0], [Y_V0])
es([XW0, Y_V0, B0], [add0])
e(add0, A0)

A1, W1, XW1, V1, Y_V1, B1, add1 = ns("A1=sigmoid() W1 ・ V1 ・ B1 +".split())
es([X, W1], [XW1])
es([Y_, V1], [Y_V1])
es([XW1, Y_V1, B1], [add1])
e(add1, A1)

A2, W2, XW2, V2, Y_V2, B2, add2 = ns("A2=tanh() W2 ・ V2 ・ B2 +".split())
es([X, W2], [XW2])
es([Y_, V2], [Y_V2])
es([XW2, Y_V2, B2], [add2])
e(add2, A2)

A3, W3, XW3, V3, Y_V3, B3, add3 = ns("A3=sigmoid() W3 ・ V3 ・ B3 +".split())
es([X, W3], [XW3])
es([Y_, V3], [Y_V3])
es([XW3, Y_V3, B3], [add3])
e(add3, A3)

C_, C_A0, A1A2, C = ns("C(t-1) * * C=add()".split())
es([A0, C_], [C_A0])
es([A1, A2], [A1A2])
es([C_A0, A1A2], [C])

tanh, Y = ns("tanh() Y(t)=mul()".split())
es([C], [tanh])
es([A3, tanh], [Y])




graph.write_png("png.png")




