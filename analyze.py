import numpy as np
from hungarian_algorithm import hungarian_algorithm, ans_calculation


def linear_assignment_problem(single_image_lst, output_writer):
    for i in range(len(single_image_lst)):
        for j in range(i, len(single_image_lst)):
            # i and j are each a cluster list for a seperate image

            img_pair_clusters_dists = find_clusters_lists_dist(
                single_image_lst[i].clusters, single_image_lst[j].clusters
            )

            # output_writer.double_print("")
            # output_writer.double_print(
            #     "Distance Between Each Cluster Of Two Images: \n"
            #     + str(img_pair_clusters_dists)
            # )

            cost_matrix = np.array(img_pair_clusters_dists)

            ans_pos = hungarian_algorithm(cost_matrix.copy())
            ans, ans_mat = ans_calculation(cost_matrix, ans_pos)

            # output_writer.double_print("")
            # output_writer.double_print(
            #     f"Linear Assignment Problem Result: {ans:.0f}\n{ans_mat}"
            # )

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


def cell_change_between_two_image(cell_id, single_image1, single_image2, min_cluster_dist):
    cell_c1 = single_image1.find_cluster_from_cell(cell_id)
    cell_c2 = single_image2.find_cluster_from_cell(cell_id)
    
    cell_clusters_dist = cell_c1.find_dist(cell_c2)

    if cell_clusters_dist > min_cluster_dist:
        changed_cluster = True
    else:
        changed_cluster = False

    return cell_clusters_dist, changed_cluster


def cell_cluster_change_proccesing(cell_id, single_image_lst, min_cluster_dist):
    counter = 0
    cluster_changes = []
    for i in range(1, len(single_image_lst)):
        cell_clusters_dist, changed_cluster = cell_change_between_two_image(
            cell_id,
            single_image_lst[0],
            single_image_lst[i],
            min_cluster_dist,
        )
        if changed_cluster:
            counter += 1
            cluster_changes.append(True)
        else:
            cluster_changes.append(False)

    cell_cluster_change_ratio = counter / (len(single_image_lst) - 1)

    return cell_cluster_change_ratio, cluster_changes


def find_cells_changed_cluster(
    num_cells, single_image_lst, min_cell_cluster_change_ratio, min_cluster_dist
):
    cell_clusters_data = {}
    likely_changed_cells = []
    for cell in single_image_lst[0].cells:
        cell_cluster_change_ratio, cluster_changes = cell_cluster_change_proccesing(
            cell.id, single_image_lst, min_cluster_dist
        )
        
        if cell_cluster_change_ratio > min_cell_cluster_change_ratio:
            likely_changed_cells.append([cell.id, cell_cluster_change_ratio])

        cell_clusters_data[cell.id] = [cluster_changes, cell_cluster_change_ratio]

    likely_changed_cells = sorted(
        likely_changed_cells, key=lambda x: x[1], reverse=True
    )

    return likely_changed_cells, cell_clusters_data


def analyze(single_image_lst, num_cells, config, output_writer):
    min_cell_cluster_change_ratio = config.min_cell_cluster_change_ratio
    min_cluster_dist = config.min_cluster_dist

    img_pair_clusters_dists = linear_assignment_problem(single_image_lst, output_writer)
    likely_changed_cells, cell_clusters_data = find_cells_changed_cluster(
        num_cells, single_image_lst, min_cell_cluster_change_ratio, min_cluster_dist
    )

    output_writer.double_print("Number of likely changed cells: " + str(len(likely_changed_cells)))
    output_writer.double_print("Likely Changed Cells:")
    [output_writer.double_print(x) for x in likely_changed_cells]
    output_writer.double_print("")
    output_writer.double_print("Cell Clusters Data:")
    [output_writer.double_print(str(x) + " " + str(cell_clusters_data[x])) for x in cell_clusters_data.keys()]
