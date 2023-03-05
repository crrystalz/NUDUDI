import os
import matplotlib.pyplot as plt
import cv2
from scipy.cluster.hierarchy import dendrogram

from cell import Cell
from cluster_cells import Cluster
from k_means_clustering import find_k_means_clusters_from_hist
from hierarchial_clustering import hierarchical_clustering


class SingleImage:
    cells = []
    kmeans_clusters = []
    hierarchial_clusters = []
    image = None
    filename = ""

    def __init__(self, image, cell_size, filename):
        (h, w) = image.shape[:2]

        hq = h // cell_size
        wq = w // cell_size

        self.cells = []

        for i in range(1, (hq + 1)):
            for j in range(1, (wq + 1)):
                id = str(i) + "-" + str(j)
                cell_image = image[
                    (i - 1) * cell_size : i * cell_size,
                    (j - 1) * cell_size : j * cell_size,
                ]

                cell = Cell(id, cell_image, cell_size)
                self.cells.append(cell)

        self.kmeans_clusters = []
        self.hierarchial_clusters = []
        self.image = image
        self.filename = filename

    def find_cell_from_id(self, cell_id):
        for cell in self.cells:
            if cell.id == cell_id:
                return cell

        print("Cell not found: " + cell_id)
        return None

    def find_cluster_from_cell(self, cluster_type, cell_id):
        if cluster_type == "kmeans":
            for cluster in self.kmeans_clusters:
                if cluster.has_cell(cell_id):
                    return cluster
        elif cluster_type == "hierarchial":
            for cluster in self.hierarchial_clusters:
                if cluster.has_cell(cell_id):
                    return cluster

        return None

    def average_color_of_cells(self):
        # find average color of each cell -> store in a list -> save the average color image to a folder called cell_avg_color_images
        pass

    def find_cell_distances(self):
        distances = {}
        min_distance = float("inf")
        max_distance = 0
        min_distance_cells = []
        max_distance_cells = []

        for i in range(len(self.cells)):
            for j in range(len(self.cells)):
                if i == j:
                    continue

                cell1 = self.cells[i]
                cell2 = self.cells[j]

                distance = cell1.find_hist_dist(cell2)

                distances[(cell1, cell2)] = distance

                if distance < min_distance:
                    min_distance = distance

                    min_distance_cells = [cell1, cell2]

                if distance > max_distance:
                    max_distance = distance

                    max_distance_cells = [cell1, cell2]

        return (
            min_distance_cells,
            min_distance,
            max_distance_cells,
            max_distance,
            distances,
        )

    def find_clusters(self, config, output_writer):
        if config.kmeans_clustering:
            predicted_labels = find_k_means_clusters_from_hist(self)

            cluster_id_to_cluster = {}
            for k in range(len(predicted_labels)):
                cluster_id = predicted_labels[k]
                if cluster_id not in cluster_id_to_cluster.keys():
                    new_cluster = Cluster(cluster_id)
                    cluster_id_to_cluster[cluster_id] = new_cluster
                    self.kmeans_clusters.append(new_cluster)

                (h, w) = self.image.shape[:2]
                num_columns = w // config.cell_size

                row = (k // num_columns) + 1
                column = (k % num_columns) + 1

                cluster_id_to_cluster[cluster_id].add_cell(
                    self.find_cell_from_id(str(row) + "-" + str(column))
                )

        if config.hierarchial_clustering:
            hierarchial_clusters_v, linkage_matrix = hierarchical_clustering(
                self, config, output_writer
            )
            for i in range(len(hierarchial_clusters_v)):
                hierarchial_cluster = Cluster(i)
                self.hierarchial_clusters.append(hierarchial_cluster)

                for cell_id in hierarchial_clusters_v[i]:
                    cell = self.find_cell_from_id(cell_id)
                    hierarchial_cluster.add_cell(cell)

            return linkage_matrix

        return None

    def output_clusters(self, linkage_matrix, config):
        filename = self.filename

        if config.kmeans_clustering:
            # Output for kmeans clustering
            dir_to_make = "output\\" + filename[0:-4] + "\\kmeans_clustering\\"
            os.mkdir(dir_to_make)
            dir_to_make = (
                "output\\" + filename[0:-4] + "\\kmeans_clustering\\clusters\\"
            )
            os.mkdir(dir_to_make)

            for cluster in self.kmeans_clusters:
                cluster_dir = (
                    "output\\"
                    + filename[0:-4]
                    + "\\kmeans_clustering\\clusters\\"
                    + str(cluster.id)
                )
                os.mkdir(cluster_dir)

                for i in range(len(cluster.cells)):
                    cv2.imwrite(
                        os.path.join(
                            cluster_dir, "cell_" + str(cluster.cells[i].id) + ".jpg"
                        ),
                        cluster.cells[i].image,
                    )

        if config.hierarchial_clustering:
            # Output for hierarchial clustering

            # Plot the dendrogram
            plt.figure()
            dendrogram(linkage_matrix, labels=[cell.id for cell in self.cells])

            plt.xticks(fontsize=4, rotation=90)

            plt.gca().margins(x=0)
            plt.gcf().canvas.draw()
            tl = plt.gca().get_xticklabels()
            maxsize = max([t.get_window_extent().width for t in tl])
            m = 0.2  # inch margin
            s = maxsize / plt.gcf().dpi * len(linkage_matrix) + 2 * m
            margin = m / plt.gcf().get_size_inches()[0]

            plt.gcf().subplots_adjust(left=margin, right=1.0 - margin)
            plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

            # plt.show()

            # Save the dendrogram
            dir_to_make = "output\\" + filename[0:-4] + "\\hierarchical_clustering\\"
            svg_dir = (
                "output\\"
                + filename[0:-4]
                + "\\hierarchical_clustering\\dendrogram.svg"
            )
            png_dir = (
                "output\\"
                + filename[0:-4]
                + "\\hierarchical_clustering\\dendrogram.png"
            )

            os.mkdir(dir_to_make)
            plt.savefig(svg_dir)
            plt.savefig(png_dir, dpi=150)

            dir_to_make = (
                "output\\" + filename[0:-4] + "\\hierarchical_clustering\\clusters\\"
            )
            os.mkdir(dir_to_make)

            clusters = self.hierarchial_clusters

            for i in range(len(clusters)):
                cluster_dir = (
                    "output\\"
                    + filename[0:-4]
                    + "\\hierarchical_clustering\\clusters\\"
                    + str(i)
                    + "\\"
                )
                os.mkdir(cluster_dir)

                for cell in clusters[i].cells:
                    cv2.imwrite(
                        os.path.join(cluster_dir, "cell_" + str(cell.id) + ".jpg"),
                        cell.image,
                    )
