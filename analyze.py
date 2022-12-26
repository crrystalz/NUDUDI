import numpy as np
from hungarian_algorithm import hungarian_algorithm, ans_calculation


def linear_assignment_problem(single_image_lst, output_writer):
    for i in range(len(single_image_lst)):
        for j in range(i, len(single_image_lst)):
            # i and j are each a cluster list for a seperate image

            img_pair_clusters_dists = find_clusters_lists_dist(
                single_image_lst[i].clusters, single_image_lst[j].clusters
            )

            output_writer.double_print("")
            output_writer.double_print(
                "Distance Between Each Cluster Of Two Images"
                + str(img_pair_clusters_dists)
            )

            cost_matrix = np.array(img_pair_clusters_dists)

            ans_pos = hungarian_algorithm(cost_matrix.copy())
            ans, ans_mat = ans_calculation(cost_matrix, ans_pos)

            output_writer.double_print("")
            output_writer.double_print(
                f"Linear Assignment Problem Result: {ans:.0f}\n{ans_mat}"
            )

    return img_pair_clusters_dists


def find_clusters_lists_dist(cluster_lst1, cluster_lst2):
    # Each is the list of clusters from two time points

    num_clusters = len(cluster_lst1)

    img_pair_clusters_dists = []
    for _ in range(num_clusters):
        img_pair_clusters_dists.append([])

    for i in range(num_clusters):
        for j in range(num_clusters):
            cluster1 = cluster_lst1[i]
            cluster2 = cluster_lst2[j]

            cluster_dist = cluster1.find_dist(cluster2)
            img_pair_clusters_dists[i].append(cluster_dist)

    return img_pair_clusters_dists


def cell_change_between_two_cluster(cell_num, cluster_lst1, cluster_lst2, threshold):
    for clusters in cluster_lst1:
        for cluster in clusters:
            if cell_num in clusters:
                cell_c1 = cluster
    for clusters in cluster_lst2:
        for cluster in clusters:
            if cell_num in clusters:
                cell_c2 = cluster

    cell_clusters_dist = cell_c1.find_dist(cell_c2)

    if cell_clusters_dist > threshold:
        changed_cluster = True
    else:
        changed_cluster = False

    return cell_clusters_dist, changed_cluster


def cell_cluster_change_proccesing(cell_num, single_image_lst, change_threshold):
    counter = 0
    cluster_changes = []
    for i in range(1, len(single_image_lst)):
        cell_clusters_dist, changed_cluster = cell_change_between_two_cluster(
            cell_num, single_image_lst[0], single_image_lst[i], change_threshold
        )
        if changed_cluster:
            counter += 1
            cluster_changes.append(True)
        else:
            cluster_changes.append(False)

    cell_cluster_change_ratio = counter / (len(single_image_lst) - 1)

    return cell_cluster_change_ratio, cluster_changes


def find_cells_changed_cluster(
    num_cells, single_image_lst, ratio_threshold, change_threshold
):
    cell_clusters_data = {}
    likely_changed_cells = []
    for cell_num in range(num_cells):
        cell_cluster_change_ratio, cluster_changes = cell_cluster_change_proccesing(
            cell_num, single_image_lst, change_threshold
        )
        if cell_cluster_change_ratio > ratio_threshold:
            likely_changed_cells.append([cell_num, cell_cluster_change_ratio])

        cell_clusters_data[cell_num] = [cluster_changes, cell_cluster_change_ratio]

    likely_changed_cells = sorted(
        likely_changed_cells, key=lambda x: x[1], reverse=True
    )

    return likely_changed_cells, cell_clusters_data


def analyze(single_image_lst, num_cells, output_writer):
    img_pair_clusters_dists = linear_assignment_problem(single_image_lst, output_writer)
    likely_changed_cells, cell_clusters_data = find_cells_changed_cluster(
        num_cells, single_image_lst, 0.5, 0.5
    )

    output_writer.double_print("Likely Changed Cells:")
    output_writer.double_print(likely_changed_cells)
    output_writer.double_print("")
    output_writer.double_print("Cell Clusters Data:")
    output_writer.double_print(cell_clusters_data)
