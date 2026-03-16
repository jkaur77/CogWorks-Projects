import networkx as nx
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import skimage.io as io
import os

from facenet_models import FacenetModel


def vectorize_photo(path):
    image = io.imread(str(path))
    if image.shape[-1] == 4:
        # Image is RGBA, where A is alpha -> transparency
        # Must make image RGB.
        image = image[..., :-1]  # png -> RGB

    return image


def proc():
    filepaths = []

    person_path = "./data/"

    filelist = os.listdir(person_path)
    for i in os.listdir(person_path):
        if i.endswith(".jpg" or ".png"):  # You could also add "and i.startswith('f')
            filepaths.append(person_path + "/" + i)

    print(filepaths)
    vectorized_images = []

    for photo_path in filepaths:
        image = vectorize_photo(photo_path)
        vectorized_images.append(image)
    model = FacenetModel()

    vectors = []
    for image in vectorized_images:
        boxes, probabilities, landmarks = model.detect(image)
        descriptions = model.compute_descriptors(image, boxes)
        for description in descriptions:
            vectors.append(description)
    return vectors



def get_cos_dist(a, b):
    result = b @ a
    return 1 - result / (np.linalg.norm(a) * np.linalg.norm(b))  # norm not normal


# list of np arrays

def generateGraph(vectors):
    threshold = 0.9  # this is the max threshold used to identify edges
    nodes = []
    matrix = []  # we'll be returning this
    # init matrix
    for i in range(len(vectors)):
        matrix.append([])
        for j in range(len(vectors)):
            matrix[i].append(0)

    for i in range(len(vectors)):
        neighbors = []
        for j in range(len(vectors)):
            if i == j:
                continue

            dist = get_cos_dist(vectors[i], vectors[j])
            if dist < threshold:
                matrix[i][j] = 1/(dist**2)
                neighbors.append(j)
        curnode = Node(i, neighbors, vectors[i])
        nodes.append(curnode)
    return matrix, nodes

class Node:
    """ Describes a node in a graph, and the edges connected
        to that node."""

    def __init__(self, ID, neighbors, descriptor, truth=None, file_path=None):
        """
        Parameters
        ----------
        ID : int
            A unique identifier for this node. Should be a
            value in [0, N-1], if there are N nodes in total.

        neighbors : Sequence[int]
            The node-IDs of the neighbors of this node.

        descriptor : numpy.ndarray
            The shape-(512,) descriptor vector for the face that this node corresponds to.

        truth : Optional[str]
            If you have truth data, for checking your clustering algorithm,
            you can include the label to check your clusters at the end.
            If this node corresponds to a picture of Ryan, this truth
            value can just be "Ryan"

        file_path : Optional[str]
            The file path of the image corresponding to this node, so
            that you can sort the photos after you run your clustering
            algorithm
        """
        self.id = ID  # a unique identified for this node - this should never change

        # The node's label is initialized with the node's ID value at first,
        # this label is then updated during the whispers algorithm
        self.label = ID

        # (n1_ID, n2_ID, ...)
        # The IDs of this nodes neighbors. Empty if no neighbors
        self.neighbors = tuple(neighbors)
        self.descriptor = descriptor

        self.truth = truth
        self.file_path = file_path


def plot_graph(graph, adj):
    """ Use the package networkx to produce a diagrammatic plot of the graph, with
    the nodes in the graph colored according to their current labels.
    Note that only 20 unique colors are available for the current color map,
    so common colors across nodes may be coincidental.
    Parameters
    ----------
    graph : Tuple[Node, ...]
        The graph to plot. This is simple a tuple of the nodes in the graph.
        Each element should be an instance of the `Node`-class.

    adj : numpy.ndarray, shape=(N, N)
        The adjacency-matrix for the graph. Nonzero entries indicate
        the presence of edges.

    Returns
    -------
    Tuple[matplotlib.fig.Fig, matplotlib.axis.Axes]
        The figure and axes for the plot."""

    g = nx.Graph()
    for n, node in enumerate(graph):
        g.add_node(n)

    # construct a network-x graph from the adjacency matrix: a non-zero entry at adj[i, j]
    # indicates that an egde is present between Node-i and Node-j. Because the edges are
    # undirected, the adjacency matrix must be symmetric, thus we only look ate the triangular
    # upper-half of the entries to avoid adding redundant nodes/edges
    g.add_edges_from(zip(*np.where(np.triu(adj) > 0)))

    # we want to visualize our graph of nodes and edges; to give the graph a spatial representation,
    # we treat each node as a point in 2D space, and edges like compressed springs. We simulate
    # all of these springs decompressing (relaxing) to naturally space out the nodes of the graph
    # this will hopefully give us a sensible (x, y) for each node, so that our graph is given
    # a reasonable visual depiction
    pos = nx.spring_layout(g)

    # make a mapping that maps: node-lab -> color, for each unique label in the graph
    color = list(iter(cm.tab20b(np.linspace(0, 1, len(set(i.label for i in graph))))))
    color_map = dict(zip(sorted(set(i.label for i in graph)), color))
    colors = [color_map[i.label] for i in graph]  # the color for each node in the graph, according to the node's label

    # render the visualization of the graph, with the nodes colored based on their labels!
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(g, pos=pos, ax=ax, nodelist=range(len(graph)), node_color=colors)
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=g.edges())
    return fig, ax