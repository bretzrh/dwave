import networkx as nx
# Create empty graph
G = nx.Graph()
# Add nodes (showing both ways to do so)
G.add_nodes_from([0,1,2], weight = -1)
# Add edges
G.add_weighted_edges_from([(0,1, 2.0),(0,2, 2.0),(1,2, 2.0)])


# Draw the graph
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
pos = nx.circular_layout(G)
#nx.draw(G,pos,with_labels=True)
nlabels=dict((n,(n,d['weight'])) for n,d in G.nodes(data=True))
#nx.draw(G,pos,labels=nlabels, node_size=2000)
elabels = nx.get_edge_attributes(G,'weight')
#nx.draw_networkx_edge_labels(G,pos,edge_labels=elabels,ax=plt.gca())
#plt.show()

# Parker's direct-to-DWave code
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
# Set Q for the problem QUBO
linear = {('x0', 'x0'): -1, ('x1', 'x1'): -1, ('x2', 'x2'): -1}
quadratic = {('x0', 'x1'): 2, ('x0', 'x2'): 2, ('x1', 'x2'): 2}
Q = dict(linear)
Q.update(quadratic)
# Minor-embed and sample 1000 times on a default D-Wave system
response = EmbeddingComposite(DWaveSampler()).sample_qubo(Q, num_reads=1000)
for datum in response.data(['sample', 'energy', 'num_occurrences']):
    print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)
