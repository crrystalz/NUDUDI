import numpy as np
from sklearn.cluster import KMeans
from cell import Cell
from cluster import Cluster


class SingleImage:
    cells = []
    clusters = []
    image = None

    def __init__(self, image, cell_size):
        (h, w) = image.shape[:2]

        hq = h // cell_size
        wq = w // cell_size

        for i in range(1, (hq + 1)):
            for j in range(1, (wq + 1)):
                id = str(i) + "-" + str(j)
                cell_image = image[
                    (i - 1) * cell_size : i * cell_size,
                    (j - 1) * cell_size : j * cell_size,
                ]

                cell = Cell(id, cell_image)
                self.cells.append(cell)

    def average_color_of_cells(self):
        # find average color of each cell
        # store in a list
        # save the average color image to a folder called cell_avg_color_images
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
            print(type(cell.histogram))
            cell_ids[x] = cell
            x += 1

        # num_clusters = 4 # turn this into a function paramter later
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # _, labels, _ = cv2.kmeans(cell_histograms, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # wcss = []
        # for i in range(1, 11):
        #     k_means = KMeans(n_clusters=i, init="k-means++", random_state=42)
        #     k_means.fit(cell_histograms)
        #     wcss.append(k_means.inertia_)

        # plt.plot(np.arange(1,11),wcss)
        # plt.xlabel('Clusters')
        # plt.ylabel('SSE')
        # plt.show()

        x = KMeans(n_clusters=4, init="k-means++", random_state=42).fit(cell_histograms)
        # x = k_means_optimum.fit(cell_histograms)
        # y = list(map(int, x))

        cluster_id_to_cluster = {}
        for k in range(len(x.labels_)):
            cluster_id = x.labels_[k]
            if cluster_id not in cluster_id_to_cluster:
                new_cluster = Cluster(cluster_id)
                cluster_id_to_cluster[cluster_id] = new_cluster
                self.clusters.append(new_cluster)
            cluster_id_to_cluster[cluster_id].add_cell(cell_ids[k])

        # ids = []

        # for k in range(len(y)):
        # k+1 = id of cell
        # y[k] = id of cluster for that cell

        # print(k + 1, y[k])

        # if y[k] not in ids:  # new cluster, y[k] is the id of the cluster
        #     ids.append(y[k])
        #     self.clusters.append(Cluster(y[k], [cell_id_to_cell[k + 1]]))

        # else:
        #     self.clusters[y[k]].add_cell(cell_id_to_cell[k + 1])
