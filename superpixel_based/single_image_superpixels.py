from skimage import io, segmentation
from skimage.future import graph
from superpixel_based.graph_merge_custom import merge_hierarchical
import numpy as np


def merge_boundary(graph, src, dst):
    """Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass


class SingleImage:
    # Original image.
    _img = None
    # superpixels computed by SLIC algorithm.
    _superpixels = None
    # Regions computed from hierarchial clustering.
    _regions = None
    # Mapping Region label to corresponding node in the RAG.
    _label_to_node = {}
    # Region Adjency Graph (RAG).
    _rag = None

    def __init__(self, img):
        self._img = img
        print("shape: ", img.shape)

    # Returns the original image.
    def image(self):
        return self._img

    # Returns an image with overlays of segments obtained from clustering.
    def image_with_regions(self):
        return segmentation.mark_boundaries(self._img, self._regions)

    # Returns an image with superpixels overlaid.
    def image_with_superpixels(self):
        return segmentation.mark_boundaries(self._img, self._superpixels)

    # Compute superpixels and then apply hierarchical merging.
    def compute(self, clustering_threshold, num_segments_for_slic):
        self._clustering_threshold = clustering_threshold
        self._num_segments_for_slic = num_segments_for_slic

        self._superpixels = segmentation.slic(
            self._img,
            start_label=1,
            compactness=10,
            n_segments=self._num_segments_for_slic,
        )

        # Compute region adjency graph.
        self._rag = graph.rag_mean_color(self._img, self._superpixels)

        # Perform standard hierarchical clustering
        self._regions = merge_hierarchical(
            self._superpixels,
            self._rag,
            thresh=self._clustering_threshold,
            rag_copy=False,
            in_place_merge=True,
            merge_func=merge_boundary,
            weight_func=graph.rag.min_weight,
        )

        for ix, n in enumerate(self._rag.nodes()):
            self._label_to_node[ix] = n

    # normal_regions is a list of (x,y)'s that user clicked on to indicate
    # that regions containing those points are not anomalous.
    def recompute(self, normal_regions):
        # map (x,y) to nodes in the RAG that must be merged.
        must_merge_nodes = []
        for x, y in normal_regions:
            # y coordinate is row, x coordinate is column in the image array.
            region_label = self._regions[(int(y), int(x))]
            node = self._label_to_node[region_label]
            print("x: ", x, " y: ", y, " region_label: ", region_label, " node: ", node)
            must_merge_nodes.append(node)

        # print("must merge: ", must_merge_nodes)
        self._regions = merge_hierarchical(
            self._superpixels,
            self._rag,
            thresh=self._clustering_threshold,
            rag_copy=False,
            in_place_merge=True,
            merge_func=merge_boundary,
            weight_func=graph.rag.min_weight,
            must_merge_nodes=must_merge_nodes,
        )

        for ix, n in enumerate(self._rag.nodes()):
            # print("ix_to_node: ", ix, n)
            self._label_to_node[ix] = n

    ## DEBUGGING FUNCTIONS ##

    # Return the number of regions.
    def num_of_labels(self):
        return len(self._label_to_node)

    def get_points_in_region(self, region_label):
        result = []
        for index in np.ndindex(self._regions.shape):
            if self._regions[index] == region_label:
                result.append(index)
        return result

    def get_node_for_point(self, x, y):
        label = self._regions[x][y]
        return self._label_to_node[label]
