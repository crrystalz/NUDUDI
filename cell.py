import cv2
import numpy as np


class Cell:
    id = 0
    image = None
    histogram = None

    def __init__(self, image, id):
        self.id = id
        self.image = image
        self.histogram = self.compute_histogram()

    def find_average_color(self):
        average_color_row = np.average(self.image, axis=0)
        average_color1 = np.average(average_color_row, axis=0)

        average_color = []
        for i in range(3):
            average_color.append(int(average_color1[i]))

        avg_color_img = np.ones((312, 312, 3), dtype=np.uint8)
        avg_color_img[:, :] = average_color1

        return average_color, avg_color_img

    def compute_histogram(self):
        image = self.image

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

    def find_hist_dist(self, other_cell):
        hist1 = self.histogram
        hist2 = other_cell.histogram
        distance = sum([(p - q) ** 2 for p, q in zip(hist1, hist2)]) ** 0.5
        return distance
