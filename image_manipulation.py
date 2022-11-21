import os
import numpy as np
import matplotlib.pyplot as plt
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
    green_hist = cv2.calcHist([image],[1],None,[16],[0,256])
    red_hist = cv2.calcHist([image],[2],None,[16],[0,256])
    blue_hist = cv2.calcHist([image],[0],None,[16],[0,256])

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
    
    # print(hist)

    return hist

def distance_proccesing():
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

            distance = sum([(p-q) ** 2 for p, q in zip(hist1, hist2)]) ** .5
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
    # print(cell_histograms)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(cell_histograms)
    print(kmeans.labels_)

def proccess_file(image):
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
    # ) = distance_proccesing()

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


directory = r"datasets\aipal-nchu_RiceSeedlingDataset\images"

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    if os.path.isfile(f):
        src_img = cv2.imread(f)

        print(f)

        proccess_file(src_img)

# dataset = pd.read_csv("roller_coasters.csv")
# X = dataset.iloc[:, [3,4]].values
# print(X)