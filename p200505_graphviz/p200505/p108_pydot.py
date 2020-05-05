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

xs0 = ns([" "] * 5)
xs1 = ns([" "] * 4)
es(xs0, xs1)

graph.write_png("png.png")




