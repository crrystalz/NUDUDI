import os
import shutil
import cv2

from utils import OutputWriter
from config import Config
from single_image import SingleImage
from temporal_analyzation import analyze
from display import x

for root, dirs, files in os.walk("output\\"):
    for f in files:
        os.unlink(os.path.join(root, f))
    for d in dirs:
        shutil.rmtree(os.path.join(root, d))

config = Config()
config.minmax_dist = False
config.hierarchial_clustering = True
config.kmeans_clustering = False
config.cell_size = 75
config.min_cell_cluster_change_ratio = 0.5
config.min_cluster_dist = 0.7
config.hierarchial_cluster_min_dist = 3085

# Optimal hierarchial cluster min dists:
# 150 - 14400

output_writer = OutputWriter("output\\output.txt")

directory = r"E:\nududi_datasets\testing"

single_image_lst = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    file_output_dir = "output\\" + filename[0:-4]
    os.mkdir(file_output_dir)

    if config.minmax_dist:
        file_output_dir = "output\\" + filename[0:-4] + "\\dist_cells"
        os.mkdir(file_output_dir)

    if os.path.isfile(f):
        src_img = cv2.imread(f)

        output_writer.double_print(f)

        single_image = SingleImage(src_img, config.cell_size, filename)

        if config.minmax_dist:
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

        linkage_matrix = single_image.find_clusters(config, output_writer)
        if config.hierarchial_clustering:
            single_image.output_clusters(linkage_matrix, config)

        x(single_image.hierarchial_clusters, single_image.image, filename)

        single_image_lst.append(single_image)

# output_writer.double_print("")
# output_writer.double_print("Analyzation of clusters through images")

# analyze(single_image_lst, len(single_image_lst[0].cells), config, output_writer)

# output_writer.close()
