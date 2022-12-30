from sklearn.cluster import KMeans
from cluster import Cluster

def find_k_means_clusters_from_hist(single_image):
        cell_histograms = []
        x = 0
        for cell in single_image.cells:
            cell_histograms.append(cell.histogram)
            x += 1

        model = KMeans(n_clusters=4, init="k-means++", random_state=42, n_init = 10)
        model.fit(cell_histograms)
        predicted_labels = model.predict(cell_histograms)

        return predicted_labels