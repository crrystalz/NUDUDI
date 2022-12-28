# Single Image
#
#
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
#
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
