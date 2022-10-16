import os
import numpy as np
import cv2

directory = r'datasets\aipal-nchu_RiceSeedlingDataset\images'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    if os.path.isfile(f):
        src_img = cv2.imread(f)

        (h, w) = src_img.shape[:2]

        cell_size = 300

        r = w % cell_size  # Remainder
        q = w // cell_size  # Quotient

        cell_colors = {}
        cell_histograms = []

        for i in range(1, (q + 1)):
            for j in range(1, (q + 1)):
                cell = src_img[(i - 1) * cell_size : i * cell_size, (j - 1) * cell_size : j * cell_size]     

                average_color_row = np.average(cell, axis=0)
                average_color = np.average(average_color_row, axis=0)
                print(average_color)
                cell_colors[str(i) + str(j)] = average_color

                d_img = np.ones((312,312,3), dtype=np.uint8)
                d_img[:,:] = average_color

                histogram, bin_edges = np.histogram(cell, bins=256, range=(0, 1))
                cell_histograms.append(histogram)

                cv2.imshow('Cell', cell)
                cv2.imshow('Average Color', d_img)
                cv2.waitKey(0)

        for i, hist1 in cell_histograms:
            for j, hist2 in cell_histograms:
                distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                print(distance)
