import numpy as np
from sklearn.cluster import KMeans
from cell import Cell
from cluster import Cluster


class SingleImage:
    cells = []
    clusters = []
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

                cell = Cell(id, cell_image)
                self.cells.append(cell)

        self.clusters = []
        self.image = image
        self.filename = filename

    def find_cluster_from_cell(self, cell_id):
        for cluster in self.clusters:
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

    def find_k_means_clusters_from_hist(self):
        cell_histograms = []
        cell_ids = {}
        x = 0
        for cell in self.cells:
            cell_histograms.append(cell.histogram)
            cell_ids[x] = cell
            x += 1

        model = KMeans(n_clusters=4, init="k-means++", random_state=42)
        model.fit(cell_histograms)
        predicted_labels = model.predict(cell_histograms)

        cluster_id_to_cluster = {}
        for k in range(len(predicted_labels)):
            cluster_id = predicted_labels[k]
            if cluster_id not in cluster_id_to_cluster.keys():
                new_cluster = Cluster(cluster_id)
                cluster_id_to_cluster[cluster_id] = new_cluster
                self.clusters.append(new_cluster)

            cluster_id_to_cluster[cluster_id].add_cell(cell_ids[k])
