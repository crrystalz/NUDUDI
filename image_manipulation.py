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
    
    min_distance_cells, min_distance, max_distance_cells, max_distance = distance_proccesing()

    print("Min Distance", min_distance)
    print("Min Distance Cells: " + str(min_distance_cells[0]) + " " + str(min_distance_cells[1]))

    min_dist_c1_name = "Min Dist Cell 1 (Cell " + str(min_distance_cells[0]) + ")"
    min_dist_c2_name = "Min Dist Cell 2 (Cell " + str(min_distance_cells[1]) + ")"

    cv2.imshow(min_dist_c1_name, cells[min_distance_cells[0] - 1])
    cv2.imshow(min_dist_c2_name, cells[min_distance_cells[1] - 1])
    cv2.waitKey(0)

    print("Max Distance", max_distance)
    print("Max Distance Cells: " + str(max_distance_cells[0]) + " " + str(max_distance_cells[1]))

    max_dist_c1_name = "Max Dist Cell 1 (Cell " + str(max_distance_cells[0]) + ")"
    max_dist_c2_name = "Max Dist Cell 2 (Cell " + str(max_distance_cells[1]) + ")"

    cv2.imshow(max_dist_c1_name, cells[max_distance_cells[0] - 1])
    cv2.imshow(max_dist_c2_name, cells[max_distance_cells[1] - 1])
    cv2.waitKey(0)

def distance_proccesing():
    distances = {}
    min_distance = float('inf')
    max_distance = 0
    min_distance_cells = []
    max_distance_cells = []

    for ij1 in cell_histograms.keys():
        for ij2 in cell_histograms.keys():
            if ij1 == ij2:
                continue

            hist1 = cell_histograms[ij1]
            hist2 = cell_histograms[ij2]

            distance = 1 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            # i = x value of cell
            # j = y value of cell

            cell1_num = int(ij1[1]) * (q-1) + int(ij1[0])

            cell2_num = int(ij2[1]) * (q-1) + int(ij2[0])

            distances[(cell1_num, cell2_num)] = distance

            if distance < min_distance:
                min_distance = distance

                min_distance_cells = [cell1_num, cell2_num]

            if distance > max_distance:
                max_distance = distance

                max_distance_cells = [cell1_num, cell2_num]

    return min_distance_cells, min_distance, max_distance_cells, max_distance

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
