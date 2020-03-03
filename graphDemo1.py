import matplotlib.pyplot as plt
from networkx import nx

nodes = 20
edges = 30

G = nx.gnm_random_graph(nodes, edges)
# some properties
print("node degree clustering")
for v in nx.nodes(G):
    print('%s %d %f' % (v, nx.degree(G, v), nx.clustering(G, v)))

# print the adjacency list
for line in nx.generate_adjlist(G):
    print(line)
nx.draw(G)
plt.show()
