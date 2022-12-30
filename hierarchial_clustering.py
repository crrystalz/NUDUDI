import numpy as np
from scipy.cluster.hierarchy import linkage


def adjacent_cell_check_distance_function(hist1, hist2):
    i1, j1 = int(hist1[0]), int(hist1[1])
    i2, j2 = int(hist2[0]), int(hist2[1])

    if abs(i1 - i2) <= 1 and abs(j1 - j2) <= 1:
        # Cells are adjacent
        return sum([(p - q) ** 2 for p, q in zip(hist1[2:], hist2[2:])]) ** 0.5

    else:
        # Cells are not adjacent
        return 999999999999


def extract_clusters(linkage_matrix, cell_ids, hierarchial_cluster_min_dist):
    # Each cell starts in its own cluster
    original_clusters = []
    for i in range(len(linkage_matrix) + 1):
        original_clusters.append([i])

    old_clusters = (
        set()
    )  # if clusters 1 and 2 get merged, then 1 and 2 are added to this set
    for (
        step_log
    ) in linkage_matrix:  # log of the stuff that happened in that step of merging
        # 0 - cluster 1 id
        # 1 - cluster 2 id
        # 3 - distance of the two clusters
        # 4 - size of the merged cluster

        distance = float(step_log[2])
        if distance > hierarchial_cluster_min_dist:
            break

        cluster1_id = int(step_log[0])
        cluster2_id = int(step_log[1])
        old_clusters.add(cluster1_id)
        old_clusters.add(cluster2_id)
        cluster1 = original_clusters[cluster1_id]
        cluster2 = original_clusters[cluster2_id]

        merged_cluster = cluster1 + cluster2
        original_clusters.append(merged_cluster)
        merged_cluster_size = int(step_log[3])
        if len(merged_cluster) != merged_cluster_size:
            print(
                "Unexpected "
                + str(len(merged_cluster))
                + " "
                + str(merged_cluster_size)
            )

    clusters = []
    for i in range(len(original_clusters)):
        if i not in old_clusters:
            clusters.append([cell_ids[x] for x in original_clusters[i]])

    return clusters


def hierarchical_clustering(single_image, config, output_writer):
    histograms = []
    cell_ids = []
    for k in range(len(single_image.cells)):
        cell = single_image.cells[k]
        cell_histogram = cell.histogram

        i, j = cell.id.split("-")
        # print(i, j)
        cell_histogram = np.insert(cell_histogram, 0, i)
        cell_histogram = np.insert(cell_histogram, 1, j)

        # output_writer.double_print("Cell ID: " + str(cell.id))
        # output_writer.double_print("Cell Histogram: " + str(cell_histogram.tolist()))

        histograms.append(cell_histogram)

        cell_ids.append(cell.id)

    linkage_matrix = linkage(
        histograms, method="single", metric=adjacent_cell_check_distance_function
    )

    clusters = extract_clusters(
        linkage_matrix, cell_ids, config.hierarchial_cluster_min_dist
    )

    return clusters, linkage_matrix
