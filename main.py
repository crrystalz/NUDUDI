import os
import numpy as np
import shutil
import cv2

from utils import OutputWriter
from config import Config
from single_image import SingleImage
from analyze import analyze


def proccess_file(image, filename, cell_size, output_writer):
    single_image = SingleImage(image, cell_size, filename)

    if q_minmax_dist:
        (
            min_distance_cells,
            min_distance,
            max_distance_cells,
            max_distance,
            distances,
        ) = single_image.find_cell_distances()

        output_writer.double_print("Min Distance " + str(min_distance))
        output_writer.double_print(
            "Min Distance Cells: "
            + str(min_distance_cells[0].id)
            + " "
            + str(min_distance_cells[1].id)
        )

        min_dist_c1_name = (
            "Min Dist Cell 1 (Cell " + str(min_distance_cells[0].id) + ")"
        )
        min_dist_c2_name = (
            "Min Dist Cell 2 (Cell " + str(min_distance_cells[1].id) + ")"
        )

        max_dist_c1_name = (
            "Max Dist Cell 1 (Cell " + str(max_distance_cells[0].id) + ")"
        )
        max_dist_c2_name = (
            "Max Dist Cell 2 (Cell " + str(max_distance_cells[1].id) + ")"
        )

        output_writer.double_print("Max Distance " + str(max_distance))
        output_writer.double_print(
            "Max Distance Cells: "
            + str(max_distance_cells[0].id)
            + " "
            + str(max_distance_cells[1].id)
        )

        dist_cell_dir = "output\\" + filename[0:-4] + "\\dist_cells"
        cv2.imwrite(
            os.path.join(dist_cell_dir, min_dist_c1_name + ".jpg"),
            min_distance_cells[0].image,
        )

        cv2.imwrite(
            os.path.join(dist_cell_dir, min_dist_c2_name + ".jpg"),
            min_distance_cells[1].image,
        )

        cv2.imwrite(
            os.path.join(dist_cell_dir, max_dist_c1_name + ".jpg"),
            max_distance_cells[0].image,
        )

        cv2.imwrite(
            os.path.join(dist_cell_dir, max_dist_c2_name + ".jpg"),
            max_distance_cells[1].image,
        )

    single_image.find_k_means_clusters_from_hist()

    for cluster in single_image.clusters:
        # print(str(cluster.id))
        cluster_dir = "output\\" + filename[0:-4] + "\\clusters\\" + str(cluster.id)
        os.mkdir(cluster_dir)

        for i in range(len(cluster.cells)):
            cv2.imwrite(
                os.path.join(
                    cluster_dir, "cell" + str(cluster.cells[i].id) + ".jpg"
                ),
                cluster.cells[i].image,
            )

    return single_image


for root, dirs, files in os.walk("output\\"):
    for f in files:
        os.unlink(os.path.join(root, f))
    for d in dirs:
        shutil.rmtree(os.path.join(root, d))

output_writer = OutputWriter("output\\output.txt")

# if input("Would you like to view the minimum and maximum distance between cells, and for those cells to be saved to the output folder? (y/n)").lower() == "y":
#     q_minmax_dist = True
# else:
#     q_minmax_dist = False
q_minmax_dist = True

directory = r"datasets\example_rostock_soda_rgb\images"

single_image_lst = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    file_output_dir = "output\\" + filename[0:-4]
    os.mkdir(file_output_dir)

    file_output_dir = "output\\" + filename[0:-4] + "\\clusters"
    os.mkdir(file_output_dir)

    if q_minmax_dist:
        file_output_dir = "output\\" + filename[0:-4] + "\\dist_cells"
        os.mkdir(file_output_dir)

    if os.path.isfile(f):
        src_img = cv2.imread(f)

        output_writer.double_print(f)

        single_image = proccess_file(src_img, filename, 300, output_writer)

        single_image_lst.append(single_image)

output_writer.double_print("")
output_writer.double_print("Analyzation of clusters through images")

config = Config(0.5, 0.7)
analyze(single_image_lst, len(single_image_lst[0].cells), config, output_writer)

output_writer.close()
