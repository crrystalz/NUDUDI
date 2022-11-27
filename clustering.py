from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def find_k_means_clusters_from_hist(cells, cell_histograms):
    wcss = []
    for i in range(1, 11):
        k_means = KMeans(n_clusters=i, init="k-means++", random_state=42)
        k_means.fit(cell_histograms)
        wcss.append(k_means.inertia_)

    # plt.plot(np.arange(1,11),wcss)
    # plt.xlabel('Clusters')
    # plt.ylabel('SSE')
    # plt.show()

    k_means_optimum = KMeans(n_clusters=3, init="k-means++", random_state=42)
    y = list(map(int, k_means_optimum.fit_predict(cell_histograms)))

    clusters_dict = {}
    for k in range(len(y)):
        if y[k] not in clusters_dict:
            clusters_dict[y[k]] = [[cells[k], k]]
        else:
            clusters_dict[y[k]].append([cells[k], k])

    clusters_lst = []

    for _ in range(len(clusters_dict)):
        clusters_lst.append([])

    return clusters_dict, clusters_lst


def find_cluster_dist(l1, l2):
    s1 = set(l1)
    s2 = set(l2)

    size_s1 = len(s1)
    size_s2 = len(s2)

    intersect = s1 & s2

    size_in = len(intersect)

    jaccard_index = size_in / (size_s1 + size_s2 - size_in)

    jaccard_dist = 1 - jaccard_index

    return jaccard_dist
