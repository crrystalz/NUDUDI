def cell_change_between_two_image(
    cell_id, single_image1, single_image2, min_cluster_dist
):
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
