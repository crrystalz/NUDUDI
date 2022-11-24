import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pandas as pd
import cv2
from sklearn.cluster import KMeans


def find_average_color(image):
    average_color_row = np.average(image, axis=0)
    average_color1 = np.average(average_color_row, axis=0)

    average_color = []
    for i in range(3):
        average_color.append(int(average_color1[i]))

    avg_color_img = np.ones((312, 312, 3), dtype=np.uint8)
    avg_color_img[:, :] = average_color1

    return average_color, avg_color_img


def compute_histogram(image):
    green_hist = cv2.calcHist([image], [1], None, [16], [0, 256])
    red_hist = cv2.calcHist([image], [2], None, [16], [0, 256])
    blue_hist = cv2.calcHist([image], [0], None, [16], [0, 256])

    green_hist = [x.tolist()[0] for x in green_hist]
    red_hist = [x.tolist()[0] for x in red_hist]
    blue_hist = [x.tolist()[0] for x in blue_hist]

    hist = []
    for k in red_hist:
        hist.append(k)
    for k in green_hist:
        hist.append(k)
    for k in blue_hist:
        hist.append(k)

    return hist


def find_hist_dist():
    distances = {}
    min_distance = float("inf")
    max_distance = 0
    min_distance_cells = []
    max_distance_cells = []

    for i in range(len(cell_histograms)):
        for j in range(1, len(cell_histograms)):
            if i == j:
                continue

            hist1 = cell_histograms[i]
            hist2 = cell_histograms[j]

            distance = sum([(p - q) ** 2 for p, q in zip(hist1, hist2)]) ** 0.5
            # distance = 1 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            distances[(i, j)] = distance

            if distance < min_distance:
                min_distance = distance

                min_distance_cells = [i, j]

            if distance > max_distance:
                max_distance = distance

                max_distance_cells = [i, j]

    return min_distance_cells, min_distance, max_distance_cells, max_distance, distances


def find_k_means_clusters_from_hist():
    wcss = []
    for i in range(1, 11):
        k_means = KMeans(n_clusters=i, init="k-means++", random_state=42)
        k_means.fit(cell_histograms)
        wcss.append(k_means.inertia_)

    # plt.plot(np.arange(1,11),wcss)
    # plt.xlabel('Clusters')
    # plt.ylabel('SSE')
    # plt.show()

    k_means_optimum = KMeans(n_clusters=2, init="k-means++", random_state=42)
    y = list(map(int, k_means_optimum.fit_predict(cell_histograms)))

    global clusters_dict
    clusters_dict = {}
    for k in range(len(y)):
        if y[k] not in clusters_dict:
            clusters_dict[y[k]] = [[cells[k], k]]
        else:
            clusters_dict[y[k]].append([cells[k], k])

    global clusters_lst
    clusters_lst = []

    for _ in range(len(clusters_dict)):
        clusters_lst.append([])
    
    return


def find_cluster_dist(l1, l2):
    # print(l1[0])
    # print(l2)
    
    s1 = set(l1)
    s2 = set(l2)
    
    size_s1 = len(s1)
    size_s2 = len(s2)
 
    intersect = s1 & s2
 
    size_in = len(intersect)
 
    jaccard_index = size_in  / (size_s1 + size_s2 - size_in)
 
    jaccard_dist = 1 - jaccard_index

    return jaccard_dist


def proccess_file(image, filename):
    (h, w) = image.shape[:2]

    cell_size = 300

    global q
    r = w % cell_size  # Remainder
    q = w // cell_size  # Quotient

    global cells
    cells = []
    global cell_colors
    cell_colors = []
    global cell_histograms
    cell_histograms = []

    for i in range(1, (q + 1)):
        for j in range(1, (q + 1)):
            cell = image[
                (i - 1) * cell_size : i * cell_size, (j - 1) * cell_size : j * cell_size
            ]
            cells.append(cell)

            cell_color, cell_avg_color_img = find_average_color(cell)
            cell_colors.append(cell_color)

            cell_histograms.append(compute_histogram(cell))

            # New cell
            # print("New cell")

    # (
    #     min_distance_cells,
    #     min_distance,
    #     max_distance_cells,
    #     max_distance,
    #     distances
    # ) = find_hist_dist()

    # print("Min Distance", min_distance)
    # print(
    #     "Min Distance Cells: "
    #     + str(min_distance_cells[0])
    #     + " "
    #     + str(min_distance_cells[1])
    # )

    # min_dist_c1_name = "Min Dist Cell 1 (Cell " + str(min_distance_cells[0]) + ")"
    # min_dist_c2_name = "Min Dist Cell 2 (Cell " + str(min_distance_cells[1]) + ")"

    # cv2.imshow(min_dist_c1_name, cells[min_distance_cells[0] - 1])
    # cv2.imshow(min_dist_c2_name, cells[min_distance_cells[1] - 1])
    # cv2.waitKey(0)

    # print("Max Distance", max_distance)
    # print(
    #     "Max Distance Cells: "
    #     + str(max_distance_cells[0])
    #     + " "
    #     + str(max_distance_cells[1])
    # )

    # max_dist_c1_name = "Max Dist Cell 1 (Cell " + str(max_distance_cells[0]) + ")"
    # max_dist_c2_name = "Max Dist Cell 2 (Cell " + str(max_distance_cells[1]) + ")"

    # cv2.imshow(max_dist_c1_name, cells[max_distance_cells[0] - 1])
    # cv2.imshow(max_dist_c2_name, cells[max_distance_cells[1] - 1])
    # cv2.waitKey(0)


    find_k_means_clusters_from_hist()

    for bucket in clusters_dict.keys():
        for cell in clusters_dict[bucket]:
            clusters_lst[bucket].append(cell[1])
        
        bucket_dir = "output\\" + filename[0:-4] + "\\" + str(bucket)
        os.mkdir(bucket_dir)

        for i in range(len(clusters_dict[bucket])):
            cv2.imwrite(
                os.path.join(bucket_dir, "cell" + str(clusters_dict[bucket][i][1]) + ".jpg"),
                clusters_dict[bucket][i][0],
            )


for root, dirs, files in os.walk("output\\"):
    for f in files:
        os.unlink(os.path.join(root, f))
    for d in dirs:
        shutil.rmtree(os.path.join(root, d))

directory = r"datasets\testing"

global_clusters_lst = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    file_output_dir = "output\\" + filename[0:-4]
    # print(file_output_dir)
    os.mkdir(file_output_dir)

    if os.path.isfile(f):
        src_img = cv2.imread(f)

        print(f)

        proccess_file(src_img, filename)

        global_clusters_lst.append(clusters_lst)

for i in range(len(global_clusters_lst)):
    for j in range(1, len(global_clusters_lst)):
        cl1 = global_clusters_lst[i]
        cl2 = global_clusters_lst[j]

        for k in range(len(cl1)):
            for l in range(1, len(cl2)):
                cluster_dist = find_cluster_dist(cl1[k], cl2[l])
                print(cluster_dist)
