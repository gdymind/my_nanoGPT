# visulize the compute graph

from graphviz import Digraph
def trace(root): # using dfs to build a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes: # only add each node once
            nodes.add(v)
            for child in v._prev: # traverse each child only once
                edges.add((child, v)) # from child to parent
                build(child)
    build(root)
    return nodes, edges
def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR means left to right
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label = "{%s | %s | data %.2f | grad %.2f}" % (n._op, n.label, n.data, n.grad), shape='record')
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)))
    return dot
