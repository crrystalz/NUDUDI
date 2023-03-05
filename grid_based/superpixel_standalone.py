from skimage import io, segmentation, filters, color
from skimage.future import graph
from skimage.future.graph import rag


def merge_boundary(graph, src, dst):
    """Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass


def compute(img, clustering_threshold, num_segments_for_slic):
    # apply SLIC algorithm to find superpixels.
    superpixels = segmentation.slic(
        img, start_label=1, compactness=10, n_segments=num_segments_for_slic
    )

    superpixels_img = segmentation.mark_boundaries(img, superpixels)

    g = graph.rag_mean_color(img, superpixels)

    # perform hierarchical clustering
    merged_superpixels = graph.merge_hierarchical(
        superpixels,
        g,
        thresh=clustering_threshold,
        rag_copy=False,
        in_place_merge=True,
        merge_func=merge_boundary,
        weight_func=rag.min_weight,
    )

    merged_superpixels_img = segmentation.mark_boundaries(img, merged_superpixels)

    return [superpixels_img, merged_superpixels_img]
