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

                distances[[cell1, cell2]] = distance

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
        for cell in self.cells:
            cell_histograms.append(cell.histogram)

        wcss = []
        for i in range(1, 11):
            k_means = KMeans(n_clusters=i, init="k-means++", random_state=42)
            k_means.fit(cell_histograms)
            wcss.append(k_means.inertia_)

        # plt.plot(np.arange(1,11),wcss)
        # plt.xlabel('Clusters')
        # plt.ylabel('SSE')
        # plt.show()

        k_means_optimum = KMeans(n_clusters=4, init="k-means++", random_state=42)
        y = list(map(int, k_means_optimum.fit_predict(cell_histograms)))

        for k in range(len(y)):
            self.clusters.append(Cluster(k, y[k]))
