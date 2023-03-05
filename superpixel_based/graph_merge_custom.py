# This file is a customized version of skimage/future/graph/graph_merge.py file
# from the skimage project.
#
# The merge_hierarchical function takes an additional argument `must_merge_nodes`
# which is a list of nodes from the RAG that must be merged even if its distance
# from its neighbors is greater than the threshold.

import numpy as np
import heapq

# Heap that uses two heaps.
# one general heap with all edges
# another heap that with only those edges whose source or target nodes must be merged.
class Heap:
    _general_edge_heap = []
    _must_merge_edge_heap = []
    _must_merge_nodes = []

    def __init__(self, must_merge_nodes):
        self._must_merge_nodes = must_merge_nodes

    def heappush(self, source_node, target_node, edge_weight, edge_data):
        # first element of the list is edge_weight so that when we
        # pop from heap, we get the edge with minimum weight.
        heap_item = [edge_weight, source_node, target_node, True]
        heapq.heappush(self._general_edge_heap, heap_item)

        # add to other heap if source or target of the edge must be merged.
        if (source_node in self._must_merge_nodes) or (
            target_node in self._must_merge_nodes
        ):
            print("heap item: ", heap_item)
            heapq.heappush(self._must_merge_edge_heap, heap_item)

        # Reference to the heap item in the graph
        edge_data["heap item"] = heap_item

    def heappop(self, thresh):
        if len(self._general_edge_heap) == 0:
            return None
        # if first edges in the general heap is not valid
        # pop it and continue.
        if not self._general_edge_heap[0][3]:
            heapq.heappop(self._general_edge_heap)
            return self.heappop(thresh)
        if self._general_edge_heap[0][0] < thresh:
            return heapq.heappop(self._general_edge_heap)
        else:
            if len(self._must_merge_edge_heap) == 0:
                return None
            return heapq.heappop(self._must_merge_edge_heap)


def _revalidate_node_edges(rag, node, heap_list):
    """Handles validation and invalidation of edges incident to a node.

    This function invalidates all existing edges incident on `node` and inserts
    new items in `heap_list` updated with the valid weights.

    rag : RAG
        The Region Adjacency Graph.
    node : int
        The id of the node whose incident edges are to be validated/invalidated
        .
    heap_list : list
        The list containing the existing heap of edges.
    """
    # networkx updates data dictionary if edge exists
    # this would mean we have to reposition these edges in
    # heap if their weight is updated.
    # instead we invalidate them

    for nbr in rag.neighbors(node):
        data = rag[node][nbr]
        try:
            # invalidate edges incident on `dst`, they have new weights
            data["heap item"][3] = False
            _invalidate_edge(rag, node, nbr)
        except KeyError:
            # will handle the case where the edge did not exist in the existing
            # graph
            pass

        wt = data["weight"]
        heap_list.heappush(node, nbr, wt, data)


def _rename_node(graph, node_id, copy_id):
    """Rename `node_id` in `graph` to `copy_id`."""

    graph._add_node_silent(copy_id)
    graph.nodes[copy_id].update(graph.nodes[node_id])

    for nbr in graph.neighbors(node_id):
        wt = graph[node_id][nbr]["weight"]
        graph.add_edge(nbr, copy_id, {"weight": wt})

    graph.remove_node(node_id)


def _invalidate_edge(graph, n1, n2):
    """Invalidates the edge (n1, n2) in the heap."""
    graph[n1][n2]["heap item"][3] = False


def merge_hierarchical(
    labels,
    rag,
    thresh,
    rag_copy,
    in_place_merge,
    merge_func,
    weight_func,
    must_merge_nodes=[],
):
    """Perform hierarchical merging of a RAG.

    Greedily merges the most similar pair of nodes until no edges lower than
    `thresh` remain.

    Parameters
    ----------
    labels : ndarray
        The array of labels.
    rag : RAG
        The Region Adjacency Graph.
    thresh : float
        Regions connected by an edge with weight smaller than `thresh` are
        merged.
    rag_copy : bool
        If set, the RAG copied before modifying.
    in_place_merge : bool
        If set, the nodes are merged in place. Otherwise, a new node is
        created for each merge..
    merge_func : callable
        This function is called before merging two nodes. For the RAG `graph`
        while merging `src` and `dst`, it is called as follows
        ``merge_func(graph, src, dst)``.
    weight_func : callable
        The function to compute the new weights of the nodes adjacent to the
        merged node. This is directly supplied as the argument `weight_func`
        to `merge_nodes`.
    must_merge_nodes:
        the nodes that need to merged with its neighbors irrespective of whether
        its distance from neighbors is greater than thresh
    Returns
    -------
    out : ndarray
        The new labeled array.

    """
    if rag_copy:
        rag = rag.copy()

    heap = Heap(must_merge_nodes)
    for n1, n2, data in rag.edges(data=True):
        heap.heappush(n1, n2, data["weight"], data)

    while True:
        e = heap.heappop(thresh)
        if e is None:
            break
        _, n1, n2, valid = e

        # Ensure popped edge is valid, if not, the edge is discarded
        if valid:
            # Invalidate all neigbors of `src` before its deleted
            # print("merging: ", n1, n2)
            for nbr in rag.neighbors(n1):
                _invalidate_edge(rag, n1, nbr)

            for nbr in rag.neighbors(n2):
                _invalidate_edge(rag, n2, nbr)

            if not in_place_merge:
                next_id = rag.next_id()
                _rename_node(rag, n2, next_id)
                src, dst = n1, next_id
            else:
                src, dst = n1, n2

            merge_func(rag, src, dst)
            new_id = rag.merge_nodes(src, dst, weight_func)
            # print("new_id: ", new_id, src, dst)
            _revalidate_node_edges(rag, new_id, heap)

    label_map = np.arange(labels.max() + 1)
    for ix, (n, d) in enumerate(rag.nodes(data=True)):
        for label in d["labels"]:
            label_map[label] = ix

    return label_map[labels]
