import cv2


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


def find_hist_dist(cell_histograms):
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
