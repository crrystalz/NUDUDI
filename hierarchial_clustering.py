import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


def adjacent_cell_check_distance_function(hist1, hist2):
    i1, j1 = int(hist1[0]), int(hist1[1])
    i2, j2 = int(hist2[0]), int(hist2[1])

    if abs(i1 - j1) <= 1 and abs(i2 - j2) <= 1:
        # Cells are adjacent
        return sum([(p - q) ** 2 for p, q in zip(hist1[2:], hist2[2:])]) ** 0.5

    else:
        # Cells are not adjacent
        return 999999999999


def hierarchical_cluster_cells(single_image):
    histograms = []
    for cell in single_image.cells:
        cell_histogram = cell.histogram

        i, j = cell.id.split("-")
        cell_histogram = np.insert(cell_histogram, 0, i)
        cell_histogram = np.insert(cell_histogram, 1, j)
        print(cell_histogram)

        histograms.append(cell_histogram)

    linkage_matrix = linkage(histograms, method = "single", 
    metric = adjacent_cell_check_distance_function)

    # print(linkage_matrix)

    # Plot the dendrogram
    plt.figure()
    dendrogram(linkage_matrix, labels=[cell.id for cell in single_image.cells])
    # plt.show()

    # Save the dendrogram
    parent_dir = "output\\" + single_image.filename[0:-4] + "\\hierarchical_clustering\\" 
    svg_dir = "output\\" + single_image.filename[0:-4] + "\\hierarchical_clustering\\dendrogram.svg" 
    png_dir = "output\\" + single_image.filename[0:-4] + "\\hierarchical_clustering\\dendrogram.png" 

    os.mkdir(parent_dir)
    plt.savefig(svg_dir)
    plt.savefig(png_dir, dpi=3000)

    # G = nx.complete_graph(10)
    # pos = nx.spring_layout(G)
    # xy = np.row_stack([point for key, point in pos.iteritems()])
    # x, y = np.median(xy, axis=0)
    # fig, ax = plt.subplots()
    # nx.draw(G, pos, with_labels=False, node_size=1)
    # ax.set_xlim(x-0.25, x+0.25)
    # ax.set_ylim(y-0.25, y+0.25)
    # plt.savefig(png_dir, bbox_inches=0, orientation='landscape', pad_inches=0.1)
