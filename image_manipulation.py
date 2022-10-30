import os
import numpy as np
import cv2

def proccess_file(image):
    (h, w) = image.shape[:2]

    cell_size = 300


    global q
    r = w % cell_size  # Remainder
    q = w // cell_size  # Quotient

    global cells
    cells = []
    global cell_colors
    cell_colors = {}
    global cell_histograms
    cell_histograms = {}

    for i in range(1, (q + 1)):
        for j in range(1, (q + 1)):
            cell = image[(i - 1) * cell_size : i * cell_size, (j - 1) * cell_size : j * cell_size]
            cells.append(cell)   

            cell_color, cell_one_color = average_color(cell)
            cell_colors[str(i) + str(j)] = cell_color

            histogram = cv2.calcHist([cell],[0],None,[256],[0,256]) 
            cell_histograms[str(i) + str(j)] = histogram

            # cv2.imshow('Cell', cell)
            # cv2.imshow('Average Color', cell_one_color)
            # cv2.waitKey(0)

    # print(cell_histograms.keys())
    
    min_distance_cells, min_distance = distance_proccesing()

    # print(len(cells))
    
    # print(min_distance_cells)

    print(min_distance)
    cv2.imshow('Cell 1', cells[int(min_distance_cells[0]) - 1])
    cv2.imshow('Cell 2', cells[int(min_distance_cells[1]) - 1])
    cv2.waitKey(0)

def distance_proccesing():
    distances = {}
    min_distance = float('inf')
    min_distance_cells = []

    for ij1 in cell_histograms.keys():
        for ij2 in cell_histograms.keys():
            if ij1 == ij2:
                continue

            hist1 = cell_histograms[ij1]
            hist2 = cell_histograms[ij2]

            distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            # i = x value of cell
            # j = y value of cell

            cell1_num = int(ij1[1]) * (q-1) + int(ij1[0])

            cell2_num = int(ij2[1]) * (q-1) + int(ij2[0])

            distances[(cell1_num, cell2_num)] = distance
            
            if distance < min_distance:
                min_distance = distance

                # print(q)

                # print(ij1[0], ij1[1])
                # print(ij2[0], ij2[1])

                min_distance_cells = [cell1_num, cell2_num]

    return min_distance_cells, min_distance

def average_color(image):
    average_color_row = np.average(image, axis=0)
    average_color = np.average(average_color_row, axis=0)
    # print(average_color)

    avg_color_img = np.ones((312,312,3), dtype=np.uint8)
    avg_color_img[:,:] = average_color
    
    return average_color, avg_color_img

directory = r'datasets\aipal-nchu_RiceSeedlingDataset\images'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    if os.path.isfile(f):
        src_img = cv2.imread(f)

        print(f)
        proccess_file(src_img)
