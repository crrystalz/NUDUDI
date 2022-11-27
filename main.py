import os
import numpy as np
import shutil
import cv2

from image_histogram import compute_histogram, find_hist_dist
from clustering import find_k_means_clusters_from_hist, find_cluster_dist
from hungarian_algorithm import hungarian_algorithm, ans_calculation


def find_average_color(image):
    average_color_row = np.average(image, axis=0)
    average_color1 = np.average(average_color_row, axis=0)

    average_color = []
    for i in range(3):
        average_color.append(int(average_color1[i]))

    avg_color_img = np.ones((312, 312, 3), dtype=np.uint8)
    avg_color_img[:, :] = average_color1

    return average_color, avg_color_img


def proccess_file(image, filename):
    (h, w) = image.shape[:2]

    cell_size = 300

    hq = h // cell_size
    wq = w // cell_size

    cells = []
    cell_colors = []
    cell_histograms = []

    for i in range(1, (hq + 1)):
        for j in range(1, (wq + 1)):
            cell = image[
                (i - 1) * cell_size : i * cell_size, (j - 1) * cell_size : j * cell_size
            ]
            cells.append(cell)

            cell_color, cell_avg_color_img = find_average_color(cell)
            cell_colors.append(cell_color)

            cell_histograms.append(compute_histogram(cell))

            # New cell
            # print("New cell")

    if q_minmax_dist:
        (
            min_distance_cells,
            min_distance,
            max_distance_cells,
            max_distance,
            distances,
        ) = find_hist_dist(cell_histograms)

        print("Min Distance", min_distance)
        print(
            "Min Distance Cells: "
            + str(min_distance_cells[0])
            + " "
            + str(min_distance_cells[1])
        )
        
        min_dist_c1_name = "Min Dist Cell 1 (Cell " + str(min_distance_cells[0]) + ")"
        min_dist_c2_name = "Min Dist Cell 2 (Cell " + str(min_distance_cells[1]) + ")"

        max_dist_c1_name = "Max Dist Cell 1 (Cell " + str(max_distance_cells[0]) + ")"
        max_dist_c2_name = "Max Dist Cell 2 (Cell " + str(max_distance_cells[1]) + ")"

        print("Max Distance", max_distance)
        print(
            "Max Distance Cells: "
            + str(max_distance_cells[0])
            + " "
            + str(max_distance_cells[1])
        )

        dist_cell_dir = "output\\" + filename[0:-4] + "\\dist_cells"
        cv2.imwrite(
            os.path.join(
                dist_cell_dir, min_dist_c1_name + ".jpg"
            ),
            cells[min_distance_cells[0] - 1],
        )

        cv2.imwrite(
            os.path.join(
                dist_cell_dir, min_dist_c2_name + ".jpg"
            ),
            cells[min_distance_cells[1] - 1],
        )

        cv2.imwrite(
            os.path.join(
                dist_cell_dir, max_dist_c1_name + ".jpg"
            ),
            cells[max_distance_cells[0] - 1],
        )

        cv2.imwrite(
            os.path.join(
                dist_cell_dir, max_dist_c2_name + ".jpg"
            ),
            cells[max_distance_cells[1] - 1],
        )


    clusters_dict, clusters_lst = find_k_means_clusters_from_hist(
        cells, cell_histograms
    )

    for bucket in clusters_dict.keys():
        for cell in clusters_dict[bucket]:
            clusters_lst[bucket].append(cell[1])

        bucket_dir = "output\\" + filename[0:-4] + "\\clusters\\" + str(bucket)
        os.mkdir(bucket_dir)

        for i in range(len(clusters_dict[bucket])):
            cv2.imwrite(
                os.path.join(
                    bucket_dir, "cell" + str(clusters_dict[bucket][i][1]) + ".jpg"
                ),
                clusters_dict[bucket][i][0],
            )

    return clusters_lst


for root, dirs, files in os.walk("output\\"):
    for f in files:
        os.unlink(os.path.join(root, f))
    for d in dirs:
        shutil.rmtree(os.path.join(root, d))

if input("Would you like to view the minimum and maximum distance between cells, and for those cells to be saved to the output folder? (y/n)").lower() == "y":
    q_minmax_dist = True
else:
    q_minmax_dist = False


directory = r"datasets\testing"

global_clusters_lst = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    file_output_dir1 = "output\\" + filename[0:-4] + "\\clusters"
    os.mkdir(file_output_dir1)

    if q_minmax_dist:
        file_output_dir2 = "output\\" + filename[0:-4] + "\\dist_cells"
        os.mkdir(file_output_dir2)

    if os.path.isfile(f):
        src_img = cv2.imread(f)

        print(f)

        clusters_lst = proccess_file(src_img, filename)

        global_clusters_lst.append(clusters_lst)

global_clusters_dists = []
for _ in range(len(global_clusters_lst)):
    global_clusters_dists.append([])

for i in range(len(global_clusters_lst)):
    for j in range(1, len(global_clusters_lst)):
        cl1 = global_clusters_lst[i]
        cl2 = global_clusters_lst[j]

        for k in range(len(cl1)):
            for l in range(1, len(cl2)):
                cluster_dist = find_cluster_dist(cl1[k], cl2[l])
                global_clusters_dists[i].append(cluster_dist)

print(global_clusters_dists)

cost_matrix = np.array(global_clusters_dists)

ans_pos = hungarian_algorithm(cost_matrix.copy())
ans, ans_mat = ans_calculation(cost_matrix, ans_pos)

print(f"Linear Assignment problem result: {ans:.0f}\n{ans_mat}")
