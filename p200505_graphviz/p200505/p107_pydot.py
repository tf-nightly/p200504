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

xs0 = ns([" "] * 3)
xs1 = ns([" "] * 4)
xs2 = ns([" "] * 4)
xs3 = ns([" "] * 4)
xs4 = ns([" "] * 3)
es(xs0, xs1)
es(xs1, xs2)
es(xs2, xs3)
es(xs3, xs4)

graph.write_png("png.png")




